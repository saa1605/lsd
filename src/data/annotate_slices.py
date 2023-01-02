import torch 
import torchvision
from typing import List, Tuple
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn as nn 
import torch.multiprocessing as mp

import clip

import numpy as np 
import cv2 
from PIL import Image, ImageDraw 

import os
import json 
from tqdm import tqdm 


class paths:
    panorama = '/data1/saaket/jpegs_manhattan_touchdown_2021/'
    pano_slice = '/data1/saaket/lsd_data/data/processed/pano_slices/'
    touchdown= '/data1/saaket/touchdown/data/'
    train = touchdown + 'train.json'
    dev = touchdown + 'dev.json'
    test = touchdown + 'test.json'
    train_pano_feature_files = [
        'image_features_sdr_pano_train_1000.pth',
        'image_features_sdr_pano_train_2000.pth',
        'image_features_sdr_pano_train_3000.pth', 
        'image_features_sdr_pano_train_4000.pth', 
        'image_features_sdr_pano_train_5000.pth',
        'image_features_sdr_pano_train_end.pth'
        ]
    dev_pano_feature_files = [
        'image_features_sdr_pano_dev.pth', 
        'image_features_sdr_pano_dev_350_700.pth', 
        'image_features_sdr_pano_dev_700_1050.pth', 
        'image_features_sdr_pano_dev_1050_end.pth'
    ]



class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        return self.model.encode_text(text)
    
class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)

class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    Args:
        tensors (tensor): Tensor containing images.
        image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> "ImageList":
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

class RPNDataset(Dataset):
    def __init__(self, image_list, transforms):
        self.image_list = image_list 
        self.transforms = transforms 
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = paths.pano_slice + f"{self.image_list[idx][:-2]}/{self.image_list[idx]}.jpg"
        pano = Image.open(image_path)

        return self.transforms(pano)

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

    
def generate_bounding_boxes(unique_pano_slices, device):
    fasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    fasterRCNN.eval() 
    fasterRCNN.to(device)


    tfms = torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float)
    ])
    
    # model, transform = clip.load('RN50', device=device)
    
    rpn_dataset = RPNDataset(unique_pano_slices, tfms)
    
    rpn_loader = DataLoader(rpn_dataset, batch_size=8, num_workers=8)
    bboxes = []
    for enum, imgData in enumerate(tqdm(rpn_loader)):
        imlist = ImageList(imgData, [tuple(i.size()[1:]) for i in imgData])
        imlist = imlist.to(device)
        features = fasterRCNN.backbone(imlist.tensors)
        with torch.no_grad():
            proposals, proposal_losses = fasterRCNN.rpn(imlist, features, )
        # props = proposals.detach().to('cpu').int()#.numpy().tolist()
        props = [p.detach().to('cpu').int().numpy().tolist() for p in proposals]
        del imlist 
        bboxes.extend(props)
    return bboxes 


def get_unique_panos(data_path):
    '''Compute unique panoramasa from data file. These panos are unique to a particular train/dev/test file'''
    panoids = []
    with open(data_path) as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            panoids.append(obj['main_pano'])
    return np.unique(panoids)

def get_pano2id(unique_panos):
    pano2id = {}
    id2pano = {}
    for i, pano in enumerate(unique_panos):
        pano2id[pano] = i 
        id2pano[i] = pano
    return pano2id, id2pano 

def touchdown_loader(data_path):
    texts, panoids = [], []
    with open(data_path) as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            panoids.append(obj['main_pano'])
            texts.append(obj['td_location_text'])
    return texts, panoids 

def compute_anchor_features(model, device, anchor_loader):
        anchor_features = []
        for anchor_batch in anchor_loader:
            with torch.no_grad():
                anchor_features_ = model.encode_image(anchor_batch.to(device))
            anchor_features.append(anchor_features_)
        
        anchor_features = torch.vstack(anchor_features)
        anchor_features /= anchor_features.norm(p=2, dim=-1, keepdim=True)
        return anchor_features

def get_image_features(unique_pano_slices, bboxes, image_encoder, transform, device, save_name):
    slice_features = []
    for idx, pid in enumerate(tqdm(unique_pano_slices)):
        image_path = paths.pano_slice + f"{pid[:-2]}/{pid}.jpg"
        perspective_image = Image.open(image_path)
    
        anchor_dataset = AnchorImageDataset(perspective_image, bboxes[idx], transform)
        anchor_loader = DataLoader(anchor_dataset, 
                    batch_size=1000, 
                    sampler=SequentialSampler(anchor_dataset),
                    pin_memory=False,
                    drop_last=False,
                    num_workers=8,)
        image_features = compute_anchor_features(image_encoder, device, anchor_loader)
        slice_features.append(image_features)
        if idx % 500 == 0:
            torch.save(torch.vstack(slice_features), save_name)
    return torch.vstack(slice_features)


def detect_regions(image_features, td_text, text_encoder, device):

    text_arr = td_text.split('.')
    if '' in text_arr:
        text_arr.remove('')
    image_features = image_features.to(device)
    tokens = clip.tokenize(text_arr).to(device)
    with torch.no_grad():
        text_features = text_encoder.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity_scores = image_features @ text_features.T 

    return similarity_scores 
    

def get_best_bboxes(image_features, bboxes, texts, pano_names, pano2id, text_encoder, device):
    best_bboxes = []
    for idx, pano_name in enumerate(tqdm(pano_names)):
        pano_id = pano2id[pano_name]
        features = image_features[pano_id]
        current_bboxes = bboxes[pano_id]
        td_text = texts[idx]
        scores = detect_regions(features, td_text, text_encoder, device)
        best_bboxes_ = [current_bboxes[i] for i in scores.argmax(dim=0).cpu().tolist()]
        best_bboxes.append(best_bboxes_)
    return best_bboxes 
    

# def detect_region(panoid, bboxes, td_text, transform, detector):

#     panorama = Image.open(paths.panorama + panoid + '.jpg')
    
#     anchor_dataset = AnchorImageDataset(panorama, bboxes, transform)
#     anchor_loader = DataLoader(anchor_dataset, 
#                 batch_size=1000, 
#                 sampler=SequentialSampler(anchor_dataset),
#                 pin_memory=False,
#                 drop_last=False,
#                 num_workers=8,)
#     image_features = compute_anchor_features(detector, device, anchor_loader)
    
#     text_arr = td_text.split('.')
#     text_arr.remove('')

#     tokens = clip.tokenize(text_arr).to(device)
#     with torch.no_grad():
#         text_features = detector.encode_text(tokens)
#         text_features /= text_features.norm(dim=-1, keepdim=True)

#     print(image_features.size(), text_features.size()) 
#     similarity_scores = image_features @ text_features.T 
    
#     return similarity_scores



def visualize_regions(bboxes, panoid):
    image = Image.open(paths.panorama + panoid + '.jpg')
    colors = ['red', 'blue', 'green']
    for cid, box in enumerate(bboxes):
        x1, y1, x2, y2 = box 
        draw = ImageDraw.Draw(image)
        draw.rectangle((x1, y1, x2, y2), outline=colors[cid])
    image.save(f'sample_image_{panoid}.jpg')


def main(model, preprocess, device, idx, unique_pano_slices, bounding_boxes, mode):
    
    image_features = get_image_features(unique_pano_slices[idx[0]:idx[1]], bounding_boxes[idx[0]:idx[1]], model, preprocess, device, f'/data2/saaket/image_features_sdr_perspective_{mode}{idx[0]}_{idx[1]}.pth')
    torch.save(image_features, f'/data2/saaket/image_features_sdr_perspective_{mode}{idx[0]}_{idx[1]}.pth')
    
    
    
    
    # Use Below Code for Panomrama Level Region Detection

    # Generate Bounding Boxes 
    # bounding_boxes = generate_bounding_boxes(unique_panos, device)
    # np.save(f'/data2/saaket/best_{mode}_boxes_pano.npy', bounding_boxes)
    
    # bounding_boxes = np.load(f'/data2/saaket/best_{mode}_boxes_pano.npy')
    
    
    
    # model, preprocess = clip.load("ViT-B/32", device=device) #Best model use ViT-B/32

    # image_features = get_image_features(unique_panos[3000:4000], bounding_boxes[3000:4000], model, preprocess, device)
    # torch.save(image_features, f'/data2/saaket/image_features_sdr_pano_{mode}_4000.pth')
    
    # prefixed = [filename for filename in os.listdir('/data2/saaket/') if filename.startswith("image_features_sdr_pano")]
    # prefixed = [ 'image_features_sdr_pano_dev.pth', 'image_features_sdr_pano_dev_350_700.pth', 'image_features_sdr_pano_dev_700_1050.pth', 'image_features_sdr_pano_dev_1050_end.pth']
    
    # image_features = [] 
    # for file in paths.dev_image_feature_files:
    #     image_features.extend(torch.load(f'/data2/saaket/{file}'), )
    # image_features = [feat.to('cpu') for feat in image_features]
    # image_features = torch.stack(image_features)

    # best_bboxes = get_best_bboxes(image_features, bounding_boxes, texts, panoids, pano2id, model, device) 
    # best_bboxes = torch.load('/data2/saaket/best_train_bboxes_ViT_full_pano.pth')
    # for i in range(10):
        # visualize_regions(boxes, panoids[i])
        # print(panoids[i], texts[i])
    # torch.save(best_bboxes, f'/data2/saaket/___best_{mode}_bboxes_ViT_full_pano.pth')
        
    # print(torch.stack(image_features).size())
    # model_text = TextCLIP(model)
    # model_image = ImageCLIP(model)
    # model_text = torch.nn.DataParallel(model_text)
    # model_image = torch.nn.DataParallel(model_image)

if __name__=='__main__':
    # Adjust Mode
    # mp.set_start_method('spawn')
    mode = 'train'
    unique_panos = get_unique_panos(paths.train)
    texts, panoids = touchdown_loader(paths.train)
    unique_pano_slices = []
    for pano in unique_panos:
        for i in range(8):
            unique_pano_slices.append(f'{pano}_{i}')
    # Pano Dicts 
    pano2id, id2pano = get_pano2id(unique_panos)
    print(len(unique_pano_slices))

    # bboxes = generate_bounding_boxes(unique_pano_slices, device)
    # np.save(f'/data2/saaket/best_{mode}_boxes_perspective.npy', bboxes)
    
    bounding_boxes = np.load(f'/data2/saaket/bounding_boxes/best_{mode}_boxes_perspective.npy')
    # Fix outlier boxes without affecting the rest of the program  
    for boxes_in_image in bounding_boxes:
        for box in boxes_in_image:
            if box[0] == box[2]:
                box[2] = box[0] + 1
            if box[1] == box[3]:
                box[3] = box[1] + 1
    num_processes = 4 
    # Device Config
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device) #Best model use ViT-B/32
    # model.share_memory()
    # processes = []
    idx = [[40, 5000], [5000, 10000], [10000, 15000], [15000, 20000]]
    # def get_area(box):
    #     x1, y1, x2, y2 = box 
    #     return (x2-x1)*(y2-y1)
    # for image_enum, boxes_in_image in tqdm(enumerate(bounding_boxes)):
    #     for box_enum, box in enumerate(boxes_in_image):
    #         if get_area(box) <= 1:
    #             print(image_enum, box_enum, box)
    # idx = [[20000,25000], [25000, 30000], [30000, 35000], [35000, 44456]]
    main(model, preprocess, device, idx[3], unique_pano_slices, bounding_boxes, mode)
    # for rank in range(num_processes):
    #     p = mp.Process(target=main, args=(model, preprocess, device, idx[rank], unique_pano_slices, bounding_boxes, mode))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    # main()