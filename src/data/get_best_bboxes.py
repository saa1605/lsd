from distutils.command.config import config
from PIL import Image, ImageDraw
import clip 
import json 
import numpy as np 
import torch 
from torchvision.ops import nms
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import cv2 
from itertools import chain, combinations
# import pandas as pd 
import torch.nn.functional as F
import random
from tqdm import tqdm 
from rpn import RegionProposalNetwork
from dataset import AnchorImageDataset
import pprint
from ml_collections import config_dict

setup_dict = {
    # paths
    'raw_data_path': '/data1/saaket/lsd_data/data/raw',
    'interim_save_path': '/data1/saaket/lsd_data/data/interim',
    'processed_save_path': '/data1/saaket/lsd_data/data/processed',
    'data_file': '/data1/saaket/lsd_data/data/raw/way_splits/test_data.json',
    'valUnseen_data_file': '/data1/saaket/lsd_data/data/raw/way_splits/valUnseen_data.json',
    'train_data_file': '/data1/saaket/lsd_data/data/raw/way_splits/train_data.json',
    'valSenn_data_file': '/data1/saaket/lsd_data/data/raw/way_splits/valSeen_data.json',
    'scan_levels_file': '/data1/saaket/lsd_data/data/raw/floorplans/scan_levels.json',
    'node2pix_file': '/data1/saaket/lsd_data/data/raw/floorplans/allScans_Node2pix.json',
    'geodistance_file': '/data1/saaket/lsd_data/data/raw/geodistance_nodes.json',
    'mesh2meters_file': '/data1/saaket/lsd_data/data/raw/floorplans/pix2meshDistance.json',
    'floorplans': '/data1/saaket/lsd_data/data/raw/floorplans',
    'figures_path': '../../reports/figures',

    # model details 
    'clip_version': 'ViT-B/32',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu', 
    'tokenizer': clip.tokenize,
    # feature extraction modes 
    'data_mode': 'test',
    'text_feature_mode': 'one_utterance', 
    'rpn_mode': 'conventional',
    'colors': [(240,0,30), (155,50,210), (255,255,25), (0,10,255), (255,170,230), (0,255,0)],
    'color_names': ['red', 'purple', 'yellow', 'blue', 'pink', 'green']

}

config = config_dict.ConfigDict(setup_dict)


pp = pprint.PrettyPrinter(indent=4)
from os.path import exists
def get_unique_scans(data_file):
    with open(data_file) as j:
        samples = json.load(j)
    unique_scans = np.empty(len(samples), dtype='<U16')
    for i, sample in enumerate(samples):
        unique_scans[i] = sample['scanName']
    return np.unique(unique_scans)

def compute_anchor_features(model, device, anchor_loader):
    anchor_features = []
    for anchor_batch in anchor_loader:
        with torch.no_grad():
            anchor_features_ = model.encode_image(anchor_batch.to(device))
        anchor_features.append(anchor_features_)
    
    anchor_features = torch.vstack(anchor_features)
    anchor_features /= anchor_features.norm(dim=-1, keepdim=True)
    return anchor_features

def compute_all_bounding_boxes(unique_scan_names, floorplan_path, scans_levels_file, rpn):
    samples = unique_scan_names
    scan_levels_file = scans_levels_file 
    with open(scan_levels_file) as j:
        scan_levels = json.load(j)
    coords_arr = []
    for i, sample in enumerate(tqdm(samples)):
        num_levels = int(scan_levels[str(sample)]["levels"])
        coords_arr.append({'scan_name': sample, 'coords': {}})
        for level in range(num_levels):
            image_path = f'{floorplan_path}/floor_{level}/{sample}_{level}.png'
            image = Image.open(image_path)
            coords, masks = rpn.get_coords_and_masks(image)
            coords_arr[i]['coords'][str(level)] = coords

    return coords_arr


def compute_text_features(data_file, model, device, mode="combined"):
    '''computes CLIP representation from dialog array. 
        for the mode argument: 
        1. combined: joins all observer utterances into one string
        2. one_utterance: considers all observer utterances separately
        3. one_sentence: considers all sentences from each observer utterance separately'''
    with open(data_file) as f:
        data = json.load(f)
    text_features = []
    text_features_arr = []
    for i, sample in enumerate(tqdm(data)):
        
        # Get the dialog array from the json dict 
        dialogs = sample["dialogArray"]
        
        # Isolate observer dialogs (dialogs in the form of: L0, O0, L1, O1 ..)
        dialogs = dialogs[1:len(dialogs):2]
        # Format dialog array into dialogs as specified by the mode
        if mode == 'combined':
            dialogs = [" ".join(dialogs).lower()]
        elif mode =='one_utterance':
            dialogs = dialogs 
        else:
            sentences = []
            for dialog in dialogs:
                sentences.extend(dialog.split("."))
            dialogs = [s.lower().strip() for s in sentences if s]
            
        tokens = clip.tokenize(dialogs, truncate=True)
#         text_features_arr.append({'scan_name': sample['scanName'], 'episode_id': sample['episodeId'], 'dialogs': dialogs})
        
        sample['processed_dialog_array'] = dialogs
        with torch.no_grad():
            text_features = model.encode_text(tokens.to(device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        sample['text_features'] = text_features
    return data 

def compute_image_features(scans, scan_bbox_arr, floorplan_path, model, device, transform):
    features = []
    for i, scan in enumerate(tqdm(scans)):
        features.append({'scanName': scan, 'features': {}})
        for level in scan_bbox_arr[i]['coords'].keys():
            image_path = f'{floorplan_path}/floor_{int(level)}/{scan}_{int(level)}.png'
            image = Image.open(image_path)
            coords = scan_bbox_arr[i]['coords'][level]
            anchor_dataset = AnchorImageDataset(image, coords, transform)
            anchor_loader = DataLoader(anchor_dataset, 
                    batch_size=15, 
                    sampler=SequentialSampler(anchor_dataset),
                    pin_memory=False,
                    drop_last=False,
                    num_workers=2,)
            
            anchor_features = compute_anchor_features(model, device, anchor_loader)
            features[i]['features'][level] = anchor_features
    return features 


def compute_scan2idx(unique_scans):
    scan2idx = {}
    for i, scan in enumerate(unique_scans):
        scan2idx[scan] = i
    return scan2idx

def get_similarity_scores(image_features_arr, text_features_arr, scan2idx):
    similarity_scores_arr = []
    for i, sample in enumerate(tqdm(text_features_arr)):
        text_features = sample['text_features']
        correct_scan = image_features_arr[scan2idx[sample['scanName']]]
        max_len = 0
        for floor in correct_scan['features'].keys():
            if len(correct_scan['features'][floor]) > max_len:
                max_len = len(correct_scan['features'][floor])
        similarity_scores = []
        for image_features in correct_scan['features'].values():
            similarity = (text_features @ image_features.T)
            similarity = F.pad(similarity, pad=(0, max_len - similarity.size(-1)), value=float('-inf'))
            similarity_scores.append(similarity)
        similarity_scores = torch.hstack(similarity_scores)
        sample['similarity_scores'] = similarity_scores
        sample['max_len'] = max_len
    return text_features_arr

def get_best_bbox(similarity_scores_arr, bbox_arr, scan2idx, mode='all_floors'):
    if mode == 'all_floors':
        for sample in tqdm(similarity_scores_arr):
            best_idx = list(sample['similarity_scores'].detach().cpu().argmax(dim=-1))
            best_idx = [idx.item() for idx in best_idx]
            best_floors = [idx // sample['max_len'] for idx in best_idx]
            best_boxes = [idx % sample['max_len'] for idx in best_idx]
            dialogs = sample['processed_dialog_array']
            scan_idx = scan2idx[sample['scanName']]
            coords = bbox_arr[scan_idx]['coords']
            sample['floors'] = {}
            sample['bboxes'] = []
            for i, floor in enumerate(best_floors):
                if floor not in sample['floors']:
                    sample['floors'].update({floor: []})
                
                sample['floors'][floor].append((i, dialogs[i], coords[str(floor)][best_boxes[i]]))
                sample['bboxes'].append(coords[str(floor)][best_boxes[i]])
#     if mode == 'correct_floor':
#         for sample in tqdm(similarity_scores_arr):
#             correct_floor = sample['finalLocation']['floor']
#             best_boxes = list(sample['similarity_scores'].detach().cpu()[:, sample['max_len']*correct_floor:sample['max_len']*(correct_floor+1)].argmax(dim=-1))
#             sample['floors'] = {correct_floor: []}
#             for i, idx in enumerate(best_boxes):
#                 sample['floors'][correct_floor].append((dialogs[i], coords[str(correct_floor)][best_boxes[i]]))
            
    return similarity_scores_arr 
        
def main(config):
    rpn = RegionProposalNetwork()
    unique_scans_val_unseen = get_unique_scans(config.data_file)
    scan_bbox_arr = compute_all_bounding_boxes(unique_scans_val_unseen, config.floorplans, config.scan_levels_file, rpn)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, transform = clip.load(config.clip_version, device=config.device)

    text_features = compute_text_features(config.data_file, model, device=config.device, mode=config.text_feature_mode)
    torch.save(text_features, f'{config.interim_save_path}/{config.rpn_mode}/normalized_text_features_{config.data_mode}_{config.text_feature_mode}.pt')
    pp.pprint(text_features[0])

    unique_scans_val_unseen = get_unique_scans(config.data_file)
    pp.pprint(unique_scans_val_unseen[0])

    scan_bbox_arr = np.load(f"{config.interim_save_path}/{config.rpn_mode}/scan_bbox_{config.data_mode}.npy", allow_pickle=True)
    pp.pprint(scan_bbox_arr[0])


    image_features = compute_image_features(unique_scans_val_unseen, scan_bbox_arr, config.floorplans, model, config.device, transform)
    torch.save(image_features, f'{config.interim_save_path}/{config.rpn_mode}/normalized_image_features_{config.data_mode}.pt')

    pp.pprint(image_features[0])

    scan2idx = compute_scan2idx(unique_scans_val_unseen)
    pp.pprint(scan2idx)


    similarity_scores_arr = get_similarity_scores(image_features, text_features, scan2idx)
    torch.save(similarity_scores_arr, f'{config.interim_save_path}/{config.rpn_mode}/similarity_scores_arr_{config.data_mode}_{config.text_feature_mode}.pt')

    pp.pprint(similarity_scores_arr[0])
    best_bbox_arr = get_best_bbox(similarity_scores_arr, scan_bbox_arr, scan2idx, mode='all_floors')
    torch.save(best_bbox_arr, f'{config.processed_save_path}/{config.rpn_mode}/best_bbox_arr_{config.data_mode}_{config.text_feature_mode}_all_floors.pt')


    pp.pprint(best_bbox_arr[0])

    print("Finished Processing Raw Data")
    
    # if not exists(f'{config.interim_save_path}/{config.rpn_mode}/normalized_text_features_{config.data_mode}_{config.text_feature_mode}.pt'):
    #     text_features = compute_text_features(config.data_file, model, device=config.device, mode=config.text_feature_mode)
        
    #     torch.save(text_features, f'{config.interim_save_path}/{config.rpn_mode}/normalized_text_features_{config.data_mode}_{config.text_feature_mode}.pt')
    # else:
    #     text_features = torch.load(f'{config.interim_save_path}/{config.rpn_mode}/normalized_text_features_{config.data_mode}_{config.text_feature_mode}.pt')
    # pp.pprint(text_features[0])

    # unique_scans_val_unseen = get_unique_scans(config.data_file)
    # pp.pprint(unique_scans_val_unseen[0])


    # if not exists(f"{config.interim_save_path}/{config.rpn_mode}/scan_bbox_{config.data_mode}.npy"):
    #     scan_bbox_arr = compute_all_bounding_boxes(unique_scans_val_unseen, config.floorplans, config.scan_levels_file, rpn)
    #     np.save(f"{config.interim_save_path}/{config.rpn_mode}/scan_bbox_{config.data_mode}.npy", scan_bbox_arr)
    # else:
    #     scan_bbox_arr = np.load(f"{config.interim_save_path}/{config.rpn_mode}/scan_bbox_{config.data_mode}.npy", allow_pickle=True)
    # pp.pprint(scan_bbox_arr[0])
    
    # if not exists(f'{config.interim_save_path}/{config.rpn_mode}/normalized_image_features_{config.data_mode}.pt'):
    #     image_features = compute_image_features(unique_scans_val_unseen, scan_bbox_arr, config.floorplans, model, config.device, transform)
    #     torch.save(image_features, f'{config.interim_save_path}/{config.rpn_mode}/normalized_image_features_{config.data_mode}.pt')
    # else:
    #     image_features = torch.load(f'{config.interim_save_path}/{config.rpn_mode}/normalized_image_features_{config.data_mode}.pt')
    # pp.pprint(image_features[0])

    # scan2idx = compute_scan2idx(unique_scans_val_unseen)
    # pp.pprint(scan2idx)
    
    # if not exists(f'{config.interim_save_path}/{config.rpn_mode}/similarity_scores_arr_{config.data_mode}_{config.text_feature_mode}.pt'):
    #     similarity_scores_arr = get_similarity_scores(image_features, text_features, scan2idx)
    #     torch.save(similarity_scores_arr, f'{config.interim_save_path}/{config.rpn_mode}/similarity_scores_arr_{config.data_mode}_{config.text_feature_mode}.pt')
    # else:
    #     similarity_scores_arr = torch.load(f'{config.interim_save_path}/{config.rpn_mode}/similarity_scores_arr_{config.data_mode}_{config.text_feature_mode}.pt')
    
    # pp.pprint(similarity_scores_arr[0])

    # if not exists(f"{config.processed_save_path}/{config.rpn_mode}/best_bbox_arr_{config.data_mode}_{config.text_feature_mode}_all_floors.pt"):
    #     best_bbox_arr = get_best_bbox(similarity_scores_arr, scan_bbox_arr, scan2idx, mode='all_floors')
    #     torch.save(best_bbox_arr, f'{config.processed_save_path}/{config.rpn_mode}/best_bbox_arr_{config.data_mode}_{config.text_feature_mode}_all_floors.pt')
    # else:
    #     best_bbox_arr = torch.load(f'{config.processed_save_path}/{config.rpn_mode}/best_bbox_arr_{config.data_mode}_{config.text_feature_mode}_all_floors.pt')
    # pp.pprint(best_bbox_arr[0])

    # print("Finished Processing Raw Data")

if __name__ == '__main__':
    main(config)