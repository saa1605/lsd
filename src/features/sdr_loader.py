import json 
import numpy as np 
import torch.nn as nn
from sdr_dataset import SDRDataset

class SDRLoader:
    def __init__(self, args):
        self.datasets = {}
        self.args = args
    def load_json_data(self, data_path):
        route_ids, panoids, centers, texts = [], [], [], []
        prefix = 'main'
        with open(data_path) as f:
            for line in f:
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
                texts.append(obj['td_location_text'])
        return route_ids, panoids, centers, texts
    
    def build_dataset(self, file):
        mode = file.split('/')[-1].split('.')[0]
        print(mode)
        route_ids, panoids, centers, texts = self.load_json_data(file)
        print("[{}]: Building dataset...".format(mode))
        
        dataset = SDRDataset(
            mode,
            self.args,
            texts,
            centers, 
            panoids,
            route_ids
        )
        self.datasets[mode] = dataset
        print("[{}]: Finish building dataset...".format(mode))