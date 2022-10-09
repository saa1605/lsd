import json 
import numpy as np 
import torch.nn as nn
from src.features.sdr_dataset import SDRDataset

class SDRLoader:
    def __init__(self, args):
        self.datasets = {}
        self.args = args
        self.vocab = Vocabulary()
        self.max_length = 0
        

    def load_json_data(self, data_path, target_path):
        route_ids, panoids, centers, texts, perspective_targets = [], [], [], [], []
        prefix = 'main'
        targets_array = np.load(target_path, allow_pickle=True)
        with open(data_path) as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                pano_type = prefix + '_pano'
                center_type = prefix + '_static_center'
                center = json.loads(obj[center_type])
                heading = prefix + '_heading'
                if center == {'x': -1,'y': -1}:
                    continue
                route_ids.append(obj['route_id'])
                panoids.append(obj['main_pano'])
                centers.append(center)
                perspective_targets.append(targets_array[idx][obj['main_pano']])
                texts.append(obj['td_location_text'])
        return route_ids, panoids, centers, perspective_targets, texts

    def build_vocab(self, texts, mode):
        '''Add words to the vocabulary'''
        ids = []
        seq_lengths = []
        for text in texts:
            line_ids = []
            words = text.lower().split()
            self.max_length = max(self.max_length, len(words))
            for word in words:
                word = self.vocab.add_word(word, mode)
                line_ids.append(self.vocab.word2idx[word])
            ids.append(line_ids)
            seq_lengths.append(len(words))
        text_ids = np.array([row + [0] * (self.max_length - len(row)) for row in ids])
        return text_ids, seq_lengths

    
    def build_dataset(self, file):
        mode = file.split('/')[-1].split('.')[0]
        mode = mode.replace("_debug", "")
        target_path = self.args.processed_save_path + f'/sdr_{mode}_perspective_targets_x_y.npy'
        
        print(mode)
        route_ids, panoids, centers, perspective_targets, texts = self.load_json_data(file, target_path)
        print("[{}]: Building dataset...".format(mode))
        texts_rnn, seq_lengths = self.build_vocab(texts, mode)
        if self.args.model == 'lingunet':
            texts = texts_rnn 

        dataset = SDRDataset(
            mode,
            self.args,
            texts,
            seq_lengths,
            centers, 
            perspective_targets,
            panoids,
            route_ids
        )
        self.datasets[mode] = dataset
        print("[{}]: Finish building dataset...".format(mode))



class Vocabulary:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}

    def add_word(self, word, mode):
        if word not in self.word2idx and mode in ('train', 'dev'):
            idx = len(self.idx2word)
            self.idx2word[idx] = word
            self.word2idx[word] = idx
            return word
        elif word not in self.word2idx and mode == 'test':
            return '<unk>'
        else:
            return word

    def __len__(self):
        return len(self.idx2word)