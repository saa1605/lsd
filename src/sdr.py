from mimetypes import init
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import tqdm
import numpy as np
import os.path
import os
import copy
import json
import sys 
import wandb
import random 
from src.config import parse_args
from src.features.loader import Loader
from src.features.sdr_loader import SDRLoader
from src.models.lingunet_model import LingUNet, load_oldArgs, convert_model_to_state
from src.utils import accuracy, distance_from_pixels, evaluate

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


class LingUNetAgent:
    def __init__(self, args):
        self.args = args
        self.device = (
            torch.device(f"cuda:{args.cuda}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.args.device = self.device
        self.loss_func = nn.KLDivLoss(reduction="batchmean") # why batchmean? 

        self.loader = None
        self.writer = None
        self.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
        if args.train and args.model_save:
            if not os.path.isdir(self.checkpoint_dir):
                print("Checkpoint directory under {}".format(self.checkpoint_dir))
                os.system("mkdir {}".format(self.checkpoint_dir))
            # self.writer = SummaryWriter(args.summary_dir + args.run_name)

        self.model = None
        self.optimizer = None
        self.args.clip_preprocess = None 

    def run_test(self):
        print("Starting Evaluation...")
        s = torch.load(self.args.eval_ckpt, map_location='cuda:0')
        self.old_args, self.rnn_args, self.state_dict, self.optimizer_state_dict = s.values()
        # self.args = load_oldArgs(self.args, oldArgs)
        self.model = LingUNet(self.args)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(self.state_dict)
        self.model = self.model.to(device=self.args.device)
        del self.state_dict, self.optimizer_state_dict, s
        with torch.no_grad():
            loss, acc0m, acc5m = self.eval_model(self.valseen_iterator, "valSeen")
        self.scores("valSeen", loss, acc0m, acc5m, 0)
        evaluate(self.args, "valSeen_data.json", self.args.run_name)
        with torch.no_grad():
            loss, acc0m, acc5m = self.eval_model(self.val_unseen_iterator, "valUnseen")
        self.scores("valUnseen", loss, acc0m, acc5m, 0)
        evaluate(self.args, "valUnseen_data.json", self.args.run_name)
        with torch.no_grad():
            loss, acc0m, acc5m = self.eval_model(self.test_iterator, "test")

    def run_epoch(
        self,
        info_elem,
        batch_texts,
        batch_target,
        batch_maps,
        batch_conversions,
        mode,
    ):
        B, num_maps, C, H, W = batch_maps.size()
        

        """ calculate loss """
        batch_target = batch_target.to(self.args.device)
        with torch.cuda.amp.autocast():
            preds = self.model(
                batch_maps.to(device=self.args.device),
                batch_texts.to(device=self.args.device),
            )
            loss = self.loss_func(preds, batch_target)
        le, ep = distance_from_pixels(
            args, preds.detach().cpu(), batch_conversions, info_elem, mode
        )
        return loss, accuracy(le, 0), accuracy(le, 5), ep

    def eval_model(self, data_iterator, mode):
        self.model.eval()
        loss, accuracy0m, accuracy5m = [], [], []
        submission = {}
        for (
            info_elem,
            texts,
            target,
            maps,
            conversions,
        ) in tqdm.tqdm(data_iterator):
            l, acc0m, acc5m, ep = self.run_epoch(
                info_elem, texts, target, maps, conversions, mode
            )
            loss.append(l.item())
            
            accuracy0m.append(acc0m)
            accuracy5m.append(acc5m)
            for i in ep:
                submission[i[0]] = {"viewpoint": i[1]}
        # print(self.args.evaluate)
        # if self.args.evaluate:
        fileName = f"{self.args.run_name}_{mode}_submission.json"
        fileName = os.path.join(self.args.predictions_dir, fileName)
        json.dump(submission, open(fileName, "w"), indent=3)
        print("submission saved at ", fileName)
        return (
            np.mean(loss),
            np.mean(np.asarray(accuracy0m)),
            np.mean(np.asarray(accuracy5m)),
        )

    def train_model(self):
        self.model.train()
        loss, accuracy0m, accuracy5m = [], [], []
        for (
            info_elem,
            texts,
            target,
            maps,
            conversions,
        ) in tqdm.tqdm(self.train_iterator):
            self.optimizer.zero_grad()
            l, acc0m, acc5m, _ = self.run_epoch(
                info_elem, texts, target, maps, conversions, mode="train"
            )
            self.scaler.scale(l).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss.append(l.item())
    
            accuracy0m.append(acc0m)
            accuracy5m.append(acc5m)

        return (
            np.mean(loss),
            np.mean(np.asarray(accuracy0m)),
            np.mean(np.asarray(accuracy5m)),
        )

    def tensorboard_writer(self, mode, loss, acc0m, acc5m, epoch):
        self.writer.add_scalar("Loss/" + mode, np.mean(loss), epoch)
        self.writer.add_scalar("Acc@0m/" + mode, acc0m, epoch)
        self.writer.add_scalar("Acc@5m/" + mode, acc5m, epoch)

    def scores(self, mode, loss, acc0m, acc5m, epoch):
        print(
            f"\t{mode} Epoch:{epoch} Loss:{loss} Acc@0m: {np.mean(acc0m)} Acc@5m: {np.mean(acc5m)}"
        )

    def run_train(self):
        assert self.args.num_lingunet_layers is not None
        rnn_args = {"input_size": len(self.loader.vocab)}

        self.model = LingUNet(args)
        # self.args, self.rnn_args, loaded_state_dict, loaded_optimizer_state_dict, encoder_state_dict = torch.load(self.args.eval_ckpt, map_location='cuda:0').values()
        # del loaded_state_dict, loaded_optimizer_state_dict
        num_params = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        print("Number of parameters:", num_params)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device=self.args.device)
        # self.model.module.clip.load_state_dict(encoder_state_dict)
        # self.model.load_state_dict(loaded_state_dict)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, eps=1e-4, weight_decay=0.001)
        # self.optimizer.load_state_dict(loaded_optimizer_state_dict)
        self.scaler = torch.cuda.amp.GradScaler(init_scale=256)
        # del loaded_state_dict, loaded_optimizer_state_dict
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2) 
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr, steps_per_epoch=len(self.train_iterator), epochs=self.args.num_epoch)

        print("Starting Training...")
        best_unseen_acc = float('-inf')
        best_model, save_path, patience = None, "", 0
        
        for epoch in range(self.args.num_epoch):
            metrics = {
                'train_loss': 0,
                'train_acc0m': 0,
                'train_acc5m': 0,
                'valSeen_loss': 0,
                'valSeen_acc0m': 0,
                'valSeen_acc5m': 0,
                'valUnseen_loss': 0,
                'valUnseen_acc0m': 0,
                'valUnseen_acc5m': 0,
            }
            loss, acc0m, acc5m = self.train_model()
            metrics['train_loss'], metrics['train_acc0m'], metrics['train_acc5m'] = loss, acc0m, acc5m
            self.scores("train", loss, acc0m, acc5m, epoch)
            with torch.no_grad():
                loss, acc0m, acc5m = self.eval_model(self.valseen_iterator, "val_seen")
            metrics['valSeen_loss'], metrics['valSeen_acc0m'], metrics['valSeen_acc5m'] = loss, acc0m, acc5m
            self.scores("val_seen", loss, acc0m, acc5m, epoch)
            with torch.no_grad():
                loss, acc0m, acc5m = self.eval_model(self.val_unseen_iterator, "val_unseen")
            metrics['valUnseen_loss'], metrics['valUnseen_acc0m'], metrics['valUnseen_acc5m'] = loss, acc0m, acc5m
            self.scores("val_unseen", loss, acc0m, acc5m, epoch)
            wandb.log(metrics)
            self.scheduler.step(loss)
            
            if acc0m > best_unseen_acc:

                if self.args.model_save:
                    save_path = os.path.join(
                        self.checkpoint_dir,
                        f"best_model_epoch_{epoch}_{acc0m}.pt",
                    )
                    state = convert_model_to_state(self.model, self.optimizer, args, rnn_args)
                    torch.save(state, save_path)
                    
                best_unseen_acc = acc0m
                patience = 0
                print("[Tune]: Best valUNseen accuracy:", best_unseen_acc)
            else:
                patience += 1
                if patience >= self.args.early_stopping:
                    break
            print("Patience:", patience)
        print(f"Best model saved at: {save_path}")

    def load_data(self):
        self.loader = SDRLoader(args)
        # self.loader.build_dataset(file="train_expanded_data.json")
        # self.loader.build_dataset(file="train_augmented_data.json")
        self.loader.build_dataset(file="../../touchdown/data/train.json")
        # self.loader.build_dataset(file="train_debug_data.json")
        self.loader.build_dataset(file="../../touchdown/data/dev.json")
        # self.loader.build_dataset(file="valSeen_debug_data.json")
        # self.loader.build_dataset(file="valUnseen_debug_data.json") 
        self.train_iterator = DataLoader(
            self.loader.datasets["train"],
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        self.dev_iterator = DataLoader(
            self.loader.datasets["dev"],
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        if self.args.evaluate:
            self.loader.build_dataset(file="../../touchdown/data/test.json")
            self.test_iterator = DataLoader(
                self.loader.datasets["test"],
                batch_size=self.args.batch_size,
                shuffle=False,
            )

    def run(self):
        self.load_data()
        print(self.args.evalute)
        if self.args.train:
            self.run_train()
        
        elif self.args.evaluate:
            self.run_test()


if __name__ == "__main__":
    args = parse_args()
    agent = LingUNetAgent(args)
    
    with wandb.init(project="LSD", name=args.run_name):
        wandb.config.update(args)
        agent.run()
    # agent.run()