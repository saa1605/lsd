from mimetypes import init
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
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
from src.sdr_config import parse_args
from src.features.sdr_loader import SDRLoader
from src.models.sdr_lingunet_model import CLIPLingUNet, LingUNet, load_oldArgs, convert_model_to_state
from src.utils import accuracy, distance_from_pixels, evaluate, distance_metric_sdr, accuracy_sdr

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
        self.loss_func = nn.KLDivLoss(reduction="batchmean") # 1.  
        self.args.num_gpus = 2 #torch.cuda.device_count()
        self.loader = None
        self.writer = None
        self.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
        if args.train and args.model_save:
            if not os.path.isdir(self.checkpoint_dir):
                print("Checkpoint directory under {}".format(self.checkpoint_dir))
                os.system("mkdir {}".format(self.checkpoint_dir))
        self.model = None
        self.optimizer = None
        self.args.clip_preprocess = None
        self.all_preds = [] 
        self.all_dists = []

    def run_test(self):
        print("Starting Evaluation...")
        s = torch.load(self.args.eval_ckpt)
        self.args, self.rnn_args, self.state_dict, self.optimizer_state_dict = s.values()
        
        if self.args.model == 'lingunet': 
            self.model = LingUNet(self.rnn_args, self.args)
        else: 
            self.model = CLIPLingUNet(self.rnn_args, self.args)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model, device_ids=[0, 1, 2])
        self.model.load_state_dict(self.state_dict)
        self.model = self.model.to(device=self.args.device)
        del self.state_dict, self.optimizer_state_dict, s
        with torch.no_grad():
            loss, mean_dists, acc40px, acc80px, acc120px = self.eval_model(self.dev_iterator, "dev")

        self.scores("dev", loss, mean_dists, acc40px, acc80px, acc120px, 0)

    def run_epoch(
        self,
        train_data
    ):
        B, num_maps, C, H, W = train_data['maps'].size()
        
        """ calculate loss """
        batch_target = train_data['target'].to(self.args.device)
        
        with torch.cuda.amp.autocast():
            preds = self.model(
                train_data['maps'].to(device=self.args.device),
                train_data['text'].to(device=self.args.device),
                train_data['seq_length']
            )
            loss = self.loss_func(preds, batch_target)
        self.all_preds.extend(preds.cpu())
 
        distances = distance_metric_sdr(preds, batch_target)
        self.all_dists.extend(distances)

        return loss, np.mean(distances), accuracy_sdr(distances, 40), accuracy_sdr(distances, 80), accuracy_sdr(distances, 120)

    def eval_model(self, data_iterator, mode):
        self.model.eval()
        loss, mean_distances, accuracy40px, accuracy80px, accuracy120px = [], [], [], [], []
        submission = {}
        for test_data in tqdm.tqdm(data_iterator):
            l, d, acc40px, acc80px, acc120px = self.run_epoch(
                test_data
            )
            
            loss.append(l.item())
            mean_distances.append(d)
            accuracy40px.append(acc40px)
            accuracy80px.append(acc80px)
            accuracy120px.append(acc120px)
        # # torch.save(self.all_preds, f'../reports/predictions/{self.args.run_name}_preds.pth')
        # torch.save(self.all_dists, f'../reports/predictions/{self.args.run_name}_dists.pth')
        return (
            np.mean(loss),
            np.mean(mean_distances),
            np.mean(np.asarray(accuracy40px)),
            np.mean(np.asarray(accuracy80px)),
            np.mean(np.asarray(accuracy120px)),
        )

    def train_model(self, data_iterator, mode):
        self.model.train()
        loss, mean_distances, accuracy40px, accuracy80px, accuracy120px = [], [], [], [], []
        for train_data in tqdm.tqdm(data_iterator):
            self.optimizer.zero_grad()
            l, d, acc40px, acc80px, acc120px= self.run_epoch(
                train_data
            )
            self.scaler.scale(l).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss.append(l.item())
            mean_distances.append(d)
            accuracy40px.append(acc40px)
            accuracy80px.append(acc80px)
            accuracy120px.append(acc120px)

        return (
            np.mean(loss),
            np.mean(mean_distances),
            np.mean(np.asarray(accuracy40px)),
            np.mean(np.asarray(accuracy80px)),
            np.mean(np.asarray(accuracy120px)),
        )

    # def tensorboard_writer(self, mode, loss, acc0m, acc5m, epoch):
    #     self.writer.add_scalar("Loss/" + mode, np.mean(loss), epoch)
    #     self.writer.add_scalar("Acc@0m/" + mode, acc0m, epoch)
    #     self.writer.add_scalar("Acc@5m/" + mode, acc5m, epoch)

    def scores(self, mode, loss, mean_dist, acc40px, acc80px, acc120px, epoch):
        print(
            f"\t{mode} Epoch:{epoch} Loss:{loss} Mean Dist:{mean_dist} Acc@40px: {np.mean(acc40px)} Acc@80px: {np.mean(acc80px)} Acc@120px {np.mean(acc120px)}"
        )

    def run_train(self):
        # assert self.args.num_lingunet_layers is not None
        rnn_args = {
            'input_size': len(self.loader.vocab),
            'embed_size': args.embed_size,
            'rnn_hidden_size': args.rnn_hidden_size,
            'num_rnn_layers': args.num_rnn_layers,
            'embed_dropout': args.embed_dropout,
            'bidirectional': args.bidirectional,
            'reduce': 'last' if not args.bidirectional else 'mean'
        }
        cnn_args = {'kernel_size': 5, 'padding': 2, 'deconv_dropout': args.deconv_dropout}
        out_layer_args = {'linear_hidden_size': args.linear_hidden_size, 'num_hidden_layers': args.num_linear_hidden_layers}

        if self.args.model == 'lingunet': 
            self.model = LingUNet(rnn_args, self.args)
        else: 
            self.model = CLIPLingUNet(rnn_args, self.args)
        # self.args, self.rnn_args, loaded_state_dict, loaded_optimizer_state_dict, encoder_state_dict = torch.load(self.args.eval_ckpt, map_location='cuda:0').values()
        # del loaded_state_dict, loaded_optimizer_state_dict
        num_params = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        print("Number of parameters:", num_params)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.args.device)
        

        # self.model.module.clip.load_state_dict(encoder_state_dict)
        # self.model.load_state_dict(loaded_state_dict)
       
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, eps=1e-4, weight_decay=0.001)
        # self.optimizer.load_state_dict(loaded_optimizer_state_dict)
        self.scaler = torch.cuda.amp.GradScaler(init_scale=256)
        # del loaded_state_dict, loaded_optimizer_state_dict
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=4) 

        print("Starting Training...")
        best_valid_acc = float('-inf')
        best_model, save_path, patience = None, "", 0
        
        for epoch in range(self.args.num_epoch):
            metrics = {
                'train_loss': 0,
                'train_mean_dist': 0,
                'train_acc40px': 0,
                'train_acc80px': 0,
                'train_acc120px': 0,
                'dev_loss': 0,
                'dev_mean_dist': 0,
                'dev_acc40px': 0,
                'dev_acc80px': 0,
                'dev_acc120px': 0,
            }
            loss, mean_dists, acc40px, acc80px, acc120px = self.train_model(self.train_iterator, "train")
            metrics['train_loss'], metrics['train_mean_dist'], metrics['train_acc40px'], metrics['train_acc80px'], metrics['train_acc_120px'] = loss, mean_dists, acc40px, acc80px, acc120px 
            self.scores("train", loss, mean_dists, acc40px, acc80px, acc120px, epoch)
            with torch.no_grad():
                loss, mean_dists, acc40px, acc80px, acc120px = self.eval_model(self.dev_iterator, "dev")
            metrics['dev_loss'], metrics['dev_mean_dist'], metrics['dev_acc40px'], metrics['dev_acc80px'], metrics['dev_acc120px'] = loss, mean_dists, acc40px, acc80px, acc120px 
            self.scores("dev", loss, mean_dists, acc40px, acc80px, acc120px, epoch)
            wandb.log(metrics)
            self.scheduler.step(loss)
            
            if acc40px > best_valid_acc:

                if self.args.model_save:
                    save_path = os.path.join(
                        self.checkpoint_dir,
                        # f"best_model_epoch_{epoch}_{acc40px}.pt",
                        "best_model.pt",
                    )
                    state = convert_model_to_state(self.model, self.optimizer, args, rnn_args)
                    print(save_path)
                    torch.save(state, save_path)
                    
                best_valid_acc = acc40px
                patience = 0
                print("[Tune]: Best dev accuracy:", best_valid_acc)
            else:
                patience += 1
                if patience >= self.args.early_stopping:
                    break
            print("Patience:", patience)
        print(f"Best model saved at: {save_path}")

    def load_data(self):
        self.loader = SDRLoader(args)
        self.loader.build_dataset(file="/data1/saaket/touchdown/data/train.json")
        self.loader.build_dataset(file="/data1/saaket/touchdown/data/dev.json")
        self.train_iterator = DataLoader(
            self.loader.datasets["train"],
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8
        )
        self.dev_iterator = DataLoader(
            self.loader.datasets["dev"],
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8
        )

        # if self.args.evaluate:
        #     self.loader.build_dataset(file="/data1/saaket/touchdown/data/test.json")
        #     self.test_iterator = DataLoader(
        #         self.loader.datasets["test"],
        #         batch_size=self.args.batch_size,
        #         shuffle=False,
        #     )

    def run(self):
        self.load_data()
        if self.args.train:
            self.run_train()
        
        elif self.args.evaluate:
            self.run_test()


if __name__ == "__main__":
    args = parse_args()
    agent = LingUNetAgent(args)
    
    with wandb.init(project="LSD", name=args.run_name, notes="lingunet with region annotation in sdr. Perspective image region annotation. object annotations, split in noun chunks, no duplicates"):
        wandb.config.update(args)
        agent.run()