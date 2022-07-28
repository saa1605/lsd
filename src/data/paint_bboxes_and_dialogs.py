from PIL import Image, ImageDraw
import clip 
import json 
import numpy as np 
import torch 
from torchvision.ops import nms
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import cv2 
from itertools import chain, combinations
import pandas as pd 
import torch.nn.functional as F
import random
from tqdm import tqdm 
import sys 
from src.config import config 
from torchvision import transforms

def paint_one_box(image, bbox, color):
        x1, y1, x2, y2 = bbox
        overlay = Image.new('RGBA', image.size, color+(0,))
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        draw.rectangle((x1, y1, x2, y2), fill=color+(50,))

        # Alpha composite these two images together to obtain the desired result.
        image = Image.alpha_composite(image, overlay)
        return image 

def paint_boxes(config, mode, text_feature_mode, rpn_mode):
    best_bboxes = torch.load('/home/saaket/embodiedAI/lsd/data/processed/{rpn_mode}/best_bbox_arr_{mode}_{text_feature_mode}_all_floors.pt')

    for item in best_bboxes:
        scan_name = item['scanName']
        for floor in item['floors'].keys():
            image = Image.open(f'{config.floorplans_dir}/floor_{floor}/{scan_name}_{floor}.png')
            for (idx, dialog, bbox) in item['floors'][floor]:
                if item['processed_dialog_array'][idx][-1] == '.':
                    item['processed_dialog_array'][idx] = item['processed_dialog_array'][idx][:-1] + f' that is in {config.color_names[idx]} color'
                else:
                   item['processed_dialog_array'][idx] = item['processed_dialog_array'][idx] + f' that is in {config.color_names[idx]} color' 
                image = paint_one_box(image, bbox, config.colors[idx])
            image.save()