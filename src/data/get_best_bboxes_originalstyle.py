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
# import spacy
# import en_core_web_md
# nlp = en_core_web_md.load()
import torch.nn.functional as F
import random
from tqdm import tqdm 
import matplotlib.pyplot as plt
from ipywidgets import interact

class RegionProposalNetwork:
    def get_coords_and_masks(self, pil_img, B=0.15, K_max_box_w=0.9, K_max_box_h=0.9,
                                K_min_box_w=0.03, K_min_box_h=0.03, iou_threshold=0.9):
        img = pil_img.copy()
        img = np.array(img)
        img = self._c_mean_shift(img)
        img = self._split_gray_img(img, n_labels=9)
        coords, masks = self._get_mixed_boxes_and_masks(
            img, B, K_max_box_w, K_max_box_h, K_min_box_w, K_min_box_h, iou_threshold
        )
        return coords, masks


    def _get_mixed_boxes_and_masks(self, image, B, K_max_box_w, K_max_box_h, K_min_box_w, K_min_box_h, iou_threshold):
        img = image.copy()
        h, w = img.shape
        max_box_w, max_box_h, min_box_w, min_box_h = w * K_max_box_w, h * K_max_box_h, w * K_min_box_w, h * K_min_box_h

        out_boxes = []
        out_masks = []

        labels = np.unique(img)
        combs = self._get_combinations(labels)

        comb_indexes = []
        for i, comb in enumerate(combs):
            n_img = np.isin(img, np.array(comb)).astype(np.uint8) * 255
            n_img = self._clear_noise(n_img)
            m_boxes = self._get_boxes_from_mask(n_img, max_box_w, max_box_h, min_box_w, min_box_h)
            out_boxes.extend(m_boxes)
            comb_indexes.extend([i] * len(m_boxes))

        comb_indexes = np.array(comb_indexes)

        boxes = torch.tensor(out_boxes, dtype=torch.float32)
        labels = torch.ones(boxes.shape[0], dtype=torch.float32)

        indexes = nms(boxes, labels, iou_threshold)

        out_boxes = boxes[indexes].numpy().astype(np.int32)
        comb_indexes = comb_indexes[indexes.numpy()]

        for (x1, y1, x2, y2), comb_index in zip(out_boxes, comb_indexes):
            comb = combs[comb_index]
            n_img = np.isin(img, np.array(comb)).astype(np.uint8) * 255
            n_img = self._clear_noise(n_img)
            mask = n_img[y1:y2, x1:x2]

            h, w = mask.shape
            h_b = int(h * B)
            w_b = int(w * B)

            mask = mask.astype(np.bool)
            if mask[:h_b, :].sum() + mask[-h_b:, :].sum() + mask[:, :w_b].sum() + \
                    mask[:, -w_b:].sum() > 4 * h * w * B * (1 - B) * 0.5:
                mask = ~mask

            mask = mask.astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=3)
            mask = mask.astype(np.bool)
            out_masks.append(mask)

        return out_boxes, out_masks
    
    def _c_mean_shift(self, image):
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.pyrMeanShiftFiltering(img, 16, 48)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)

    def _split_gray_img(self, image, n_labels=9):
        img = image.copy()
        step = 255 // n_labels
        t = list(np.arange(0, 255, step)) + [255]
        for i, (t1, t2) in enumerate(zip(t[:-1], t[1:])):
            img[(img >= t1) & (img < t2)] = t1
        return img

    @staticmethod
    def _get_combinations(array):
        combs = list(chain(*map(lambda x: combinations(array, x), range(0, len(array)+1))))
        return combs[1:]

    @staticmethod
    def _get_boxes_from_mask(mask, max_box_w, max_box_h, min_box_w, min_box_h):
        boxes = []
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(c) for c in contours]
        for i, (x, y, w, h) in enumerate(bboxes):
            if (w < max_box_w and h < max_box_h) and (w > min_box_w and h > min_box_h):
                boxes.append([x, y, x + w, y + h])
        return boxes

    @staticmethod
    def _clear_noise(image):
        img = image.copy()
        e_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        erose = cv2.morphologyEx(img, cv2.MORPH_ERODE, e_kernel)
        d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.morphologyEx(erose, cv2.MORPH_DILATE, d_kernel)
        return dilate

    @staticmethod
    def _get_nearest_box_and_mask(box, gt_boxes, gt_masks):
        return sorted(zip(gt_boxes, gt_masks), key=lambda x: sum([abs(x[0][i] - box[i]) for i in range(4)]))[0]

class AnchorImageDataset(Dataset):
    def __init__(self, image, coords, transforms):
        self.image = image.copy()
        self.coords = coords
        self.transforms = transforms

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        return self.transforms(self.image.crop(coord))


def compute_anchor_features(model, anchor_loader):
    anchor_features = []
    for anchor_batch in anchor_loader:
        with torch.no_grad():
            anchor_features_ = model.encode_image(anchor_batch.to(device))
        anchor_features.append(anchor_features_)
    
    anchor_features = torch.vstack(anchor_features)
    anchor_features /= anchor_features.norm(dim=-1, keepdim=True)
    return anchor_features

def compute_all_bounding_boxes(unique_scan_names, scans_levels_file, rpn):

    samples = unique_scan_names
    scan_levels_file = scans_levels_file 
    with open(scan_levels_file) as j:
        scan_levels = json.load(j)
    coords_arr = []
    for i, sample in enumerate(tqdm(samples)):
        num_levels = int(scan_levels[str(sample)]["levels"])
        coords_arr.append({'scan_name': sample, 'coords': {}})
        for level in range(num_levels):
            image_path = f'../input/way-led/floorplans/floor_{level}/{sample}_{level}.png'
            image = Image.open(image_path)
            coords, masks = rpn.get_coords_and_masks(image)
            coords_arr[i]['coords'][str(level)] = coords

    return coords_arr

def get_unique_scans(data_file):
    with open(data_file) as j:
        samples = json.load(j)
    unique_scans = np.empty(len(samples), dtype='<U16')
    for i, sample in enumerate(samples):
        unique_scans[i] = sample['scanName']
    return np.unique(unique_scans)

def compute_text_features(data_file, model, mode="combined"):
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
        elif mode =='one_sentence':
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


# This is new version 
def compute_image_features(scans, scan_bbox_arr, model, transform):
    features = []
    for i, scan in enumerate(tqdm(scans)):
        features.append({'scanName': scan, 'features': {}})
        for level in scan_bbox_arr[i]['coords'].keys():
            image_path = f'../input/way-led/floorplans/floor_{int(level)}/{scan}_{int(level)}.png'
            image = Image.open(image_path)
            coords = scan_bbox_arr[i]['coords'][level]
            anchor_dataset = AnchorImageDataset(image, coords, transform)
            anchor_loader = DataLoader(anchor_dataset, 
                    batch_size=15, 
                    sampler=SequentialSampler(anchor_dataset),
                    pin_memory=False,
                    drop_last=False,
                    num_workers=2,)
            
            anchor_features = compute_anchor_features(model, anchor_loader)
            features[i]['features'][level] = anchor_features
    return features 

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
    
    best_bboxes = []
    if mode == 'all_floors':
        for sample in tqdm(similarity_scores_arr):
            best_idx = list(sample['similarity_scores'].detach().cpu().argmax(dim=-1))
            best_idx = [idx.item() for idx in best_idx]
            best_floors = [idx // sample['max_len'] for idx in best_idx]
            best_boxes = [idx % sample['max_len'] for idx in best_idx]
            dialogs = sample['processed_dialog_array']
            scan_idx = scan2idx[sample['scanName']]
            coords = bbox_arr[scan_idx]
            sample['floors'] = {}
            for i, floor in enumerate(best_floors):
                if floor not in sample['floors']:
                    sample['floors'].update({floor: []})
                sample['floors'][floor].append((dialogs[i], coords['coords'][str(floor)][best_boxes[i]]))
            sample['bboxes'] = best_boxes
            sample['floors']
#     if mode == 'correct_floor':
#         for sample in tqdm(similarity_scores_arr):
#             correct_floor = sample['finalLocation']['floor']
#             best_boxes = list(sample['similarity_scores'].detach().cpu()[:, sample['max_len']*correct_floor:sample['max_len']*(correct_floor+1)].argmax(dim=-1))
#             sample['floors'] = {correct_floor: []}
#             for i, idx in enumerate(best_boxes):
#                 sample['floors'][correct_floor].append((dialogs[i], coords[str(correct_floor)][best_boxes[i]]))
            
    return similarity_scores_arr 
        
def compute_scan2idx(unique_scans):
    scan2idx = {}
    for i, scan in enumerate(unique_scans):
        scan2idx[scan] = i
    return scan2idx

def main(data_mode):
    with open(f'/data1/saaket/lsd_data/data/raw/way_splits/{data_mode}_data.json') as f:
        data = json.load(f)
    unique_scans = get_unique_scans(f'/data1/saaket/lsd_data/data/raw/way_splits/{data_mode}_data.json')
    scan_levels_file = f'/data1/saaket/lsd_data/data/raw/floorplans/scan_levels.json'
    rpn = RegionProposalNetwork()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, transform = clip.load("RN50", device=device)
    
    scan_bbox_arr = compute_all_bounding_boxes(unique_scans, scan_levels_file, rpn)

    np.save(f"/data2/saaket/lsd_data/scan_bbox_{data_mode}.npy", scan_bbox_arr)

    text_features = compute_text_features('../input/way-led/way_splits/valUnseen_data.json', model, mode='one_utterance')

    image_features = compute_image_features(unique_scans, scan_bbox_arr, model, transform)

    torch.save(image_features, f'/data2/saaket/lsd_data/image_features_{data_mode}.pt')

    scan2idx = compute_scan2idx(unique_scans)

    similarity_scores_arr = get_similarity_scores(image_features, text_features, scan2idx)

    best_bbox_arr = get_best_bbox(similarity_scores_arr, scan_bbox_arr, scan2idx, mode='all_floors')

    torch.save(best_bbox_arr, f'/data2/saaket/lsd_data/best_bbox_arr_{data_mode}_one_utterance_all_floors.pt')

if __name__ == '__main__':
    main('valUnseen')
