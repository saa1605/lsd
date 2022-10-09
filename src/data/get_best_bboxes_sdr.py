import os
import json 
import random
from tqdm import tqdm 
import pprint
from joblib import Parallel, delayed 
import glob 
import time 

import cv2 
from cv2 import transform
from PIL import Image, ImageDraw
import numpy as np 

import torch.nn as nn 
import torchvision 
import torch.nn.functional as F
import torch 
import clip 
from torchvision.ops import nms
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from dataset import AnchorImageDatasetSDR, ImageList, RPNDataset
from models_clip import TextCLIP, ImageCLIP

# Paths and Constants 
paths = {
    'train_bboxes': '/data1/saaket/lsd_data/data/interim/sdr/rcnn/train_bboxes.npy',
    'dev_bboxes': '/data1/saaket/lsd_data/data/interim/sdr/rcnn/dev_bboxes.npy',
    'test_bboxes': '/data1/saaket/lsd_data/data/interim/sdr/rcnn/test_bboxes.npy',
    'train_perspective_image_clip_features': '/data1/saaket/lsd_data/data/processed/sdr/rcnn/train_image_features.npy',
    'dev_perspective_image_clip_features': '/data1/saaket/lsd_data/data/processed/sdr/rcnn/dev_image_features.npy',
    'test_perspective_image_clip_features': '/data1/saaket/lsd_data/data/processed/sdr/rcnn/test_image_features.npy',

}

NUM_PANO_SLICES = 8
PANO_SLICE_DIRECTORY = '/data1/saaket/lsd_data/data/processed/pano_slices'
TRAIN_PATH = '/data1/saaket/touchdown/data/train.json'
TRAIN_BOX_PATH = '../src/data/sdr_bboxes_train.npy'
# one function for each of the things 

# get unique panorama ids 
def get_unique_panos(data_path):
    panoids = []
    with open(data_path) as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            panoids.append(obj['main_pano'])
    return np.unique(panoids)

# opens an image and finds potential bboxes using a region proposal network 
def generate_bounding_boxes(unique_panos, device):
    fasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    fasterRCNN.eval() 
    fasterRCNN.to(device)


    tfms = torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float)
    ])
    
    # model, transform = clip.load('RN50', device=device)
    
    rpn_dataset = RPNDataset(unique_panos, tfms)
    
    rpn_loader = DataLoader(rpn_dataset, batch_size=16, num_workers=2)
    bboxes = []
    for enum, imgData in enumerate(tqdm(rpn_loader)):
        imlist = ImageList(imgData, [tuple(i.size()[1:]) for i in imgData])
        imlist = imlist.to(device)
        features = fasterRCNN.backbone(imlist.tensors)
        proposals, proposal_losses = fasterRCNN.rpn(imlist, features, )
        # props = proposals.detach().to('cpu').int()#.numpy().tolist()
        props = [p.detach().to('cpu').int().numpy().tolist() for p in proposals]
        
        bboxes.extend(props)
    return bboxes 

# takes the image and the bboxes and computes image features for each bounding box 
def compute_anchor_features(model, device, anchor_loader):
    anchor_features = []
    for anchor_batch in anchor_loader:
        with torch.no_grad():
            anchor_features_ = model(anchor_batch.to(device))
        anchor_features.append(anchor_features_)
    
    anchor_features = torch.vstack(anchor_features)
    anchor_features /= anchor_features.norm(dim=-1, keepdim=True)
    return anchor_features


def compute_image_features(unique_panos, coords, model_image, device, transforms):
    anchor_dataset = AnchorImageDatasetSDR(unique_panos, transforms, coords)
    features = []
    anchor_loader = DataLoader(anchor_dataset, 
                    batch_size=2, 
                    sampler=SequentialSampler(anchor_dataset),
                    pin_memory=False,
                    drop_last=False,
                    num_workers=2,)
    for i, data in enumerate(tqdm(anchor_loader)):
        bs, num_bboxes, C, H, W = data.size()
        data = data.view(bs*num_bboxes, C, H, W)
        feature_computation_start = time.time()
        with torch.no_grad():
            feat = model_image(data.to(device))
        feature_computation_end = time.time()
        print("clip compute time: ", (feature_computation_end - feature_computation_start))
        features.extend(feat)
        if i == 15:
            break
    return torch.stack(features)
# iterates through the touchdown data and computes text features for each phrase/sentence in td_location_text  
# computes similarity scores for each ( image - text-set ) features 
# calculates the the most likely bounding box for each text 










# def compute_best_bboxes(data_path, coords, model_image, model_text, device, transform):
#     samples = []
#     with open(data_path) as f:
#         for line in f:
#             obj = json.loads(line)
#             samples.append(obj)
#     best_boxes = []
#     for idx, obj in enumerate(tqdm(samples)):
#         panoid = obj['main_pano']
#         td_location_text = obj['td_location_text']
#         pano_slice_features = []
#         for jdx, pano_slice in enumerate(os.listdir(PANO_SLICE_DIRECTORY + '/' + panoid)):
#             pano_slice_image = Image.open(PANO_SLICE_DIRECTORY + '/' + panoid + '/' + pano_slice)
#             anchor_dataset = AnchorImageDataset(pano_slice_image, coords[8*idx + jdx], transform)
#             anchor_loader = DataLoader(anchor_dataset, 
#                     batch_size=1000, 
#                     sampler=SequentialSampler(anchor_dataset),
#                     pin_memory=False,
#                     drop_last=False,
#                     num_workers=8,)
            
#             anchor_features = compute_anchor_features(model_image, device, anchor_loader) 
#             pano_slice_features.append(anchor_features)
        
#         image_features = torch.vstack(pano_slice_features)
#         tokens = clip.tokenize(td_location_text.split('.'), truncate=True).to(device)
#         with torch.no_grad():
#             text_features = model_text(tokens)
#         similarity = text_features @ image_features.T
#         similarity = similarity.detach().cpu()
#         best_bboxes = similarity.argmax(dim=-1)
#         best_boxes.append(best_bboxes.tolist())
#         del tokens, anchor_features, text_features
#     return best_boxes 



def main():
    train_unique_panos = get_unique_panos('/data1/saaket/touchdown/data/train.json')
    dev_unique_panos = get_unique_panos('/data1/saaket/touchdown/data/dev.json')
    test_unique_panos = get_unique_panos('/data1/saaket/touchdown/data/test.json')
    pano_slice_directory = '/data1/saaket/lsd_data/data/processed/pano_slices'
    device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("RN50",device=device, jit=False) #Best model use ViT-B/32
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text, device_ids=[5, 6, 7])
    model_image = torch.nn.DataParallel(model_image, device_ids=[5, 6, 7])
    # train_bboxes = generate_bounding_boxes(train_unique_panos, device)
    # dev_bboxes = generate_bounding_boxes(train_unique_panos, device)
    train_bboxes = np.load('sdr_bboxes_train.npy', allow_pickle=True)
    # bboxes = compute_best_bboxes(TRAIN_PATH, train_bboxes, model_image, model_text, device, preprocess)
    # np.save('/data2/saaket/best_train_bboxes.npy', bboxes)
    feat = compute_image_features(train_unique_panos, train_bboxes, model_image, device, preprocess)
    print(feat.size())
    # print(dset[0].size())
    # np.save('sdr_bboxes.npy', bboxes)
    # bboxes = np.load('sdr_bboxes.npy', allow_pickle=True)
    
    
    

    # print(bboxes)
    # print(len(bboxes))
    # coords_arr = compute_all_bounding_boxes(pano_slice_directory, rpn)
    # np.save('/data1/saaket/lsd_data/data/processed/sdr_annotation_data', coords_arr)

    # feat = compute_image_features(pano_slice_directory, rpn, model, device, transform)
    # print(feat.size())

if __name__ == '__main__':
    main()


