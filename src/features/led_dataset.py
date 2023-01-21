from random import gauss
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn as nn 
import numpy as np
from PIL import Image, ImageDraw
import copy
import json
import src.models.clip as clip 
# from skimage.draw import polygon2mask, polygon
import os 
import sys 
import cv2 

class LEDDataset(Dataset):
    def __init__(
        self,
        mode,
        args,
        texts,
        mesh_conversions,
        locations,
        viewPoint_location,
        dialogs,
        scan_names,
        levels,
        annotation_ids,
    ):
        self.mode = mode
        self.args = args
        self.texts = texts
        self.mesh_conversions = mesh_conversions
        self.locations = locations
        self.viewPoint_location = viewPoint_location
        self.dialogs = dialogs
        self.scan_names = scan_names
        self.levels = levels
        self.annotation_ids = annotation_ids
        self.mesh2meters = json.load(open(args.image_dir + "pix2meshDistance.json"))
        # if self.mode == 'train':
            # self.mode = 'train_augmented'
        # self.bbox_data = torch.load(f'{args.processed_save_path}/{args.rpn_mode}/best_bbox_arr_{self.mode}_{args.text_feature_mode}_all_floors.pt')
        self.bbox_data = torch.load(f'/data2/saaket/lsd_data/bboxes2/best_bbox_arr_{self.mode}_{args.text_feature_mode}_all_floors.pt')
        self.image_size = [
            3,
            int(700 * self.args.ds_percent),
            int(1200 * self.args.ds_percent),
        ]
        self.region_size = [
            3,
            # int(( 700 * self.args.ds_percent ) / 2),
            # int(( 1200 * self.args.ds_percent ) / 3)
            224, 
            224
        ]

        self.preprocess_data_aug = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5, hue=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073, 0.555], # [0.485, 0.456, 0.406, 0.555],
                    std= [0.26862954, 0.26130258, 0.27577711, 0.222]# [0.229, 0.224, 0.225, 0.222],
                ),
            ]
        )
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073, 0.555], # [0.485, 0.456, 0.406, 0.555],
                    std= [0.26862954, 0.26130258, 0.27577711, 0.222]# [0.229, 0.224, 0.225, 0.222],
                ),
            ]
        )

    def gather_all_floors(self, index):
        all_maps = torch.zeros(
            self.args.max_floors,
            self.image_size[0],
            self.image_size[1],
            self.image_size[2],
        )
        all_conversions = torch.zeros(self.args.max_floors, 1)
        sn = self.scan_names[index]
        floors = self.mesh2meters[sn].keys()
        bboxes_on_floors = self.bbox_data[index]['floors']
        
        for enum, f in enumerate(floors):
            img = Image.open(
                "{}floor_{}/{}_{}.png".format(self.args.image_dir, f, sn, f)
            )
            if int(f) in bboxes_on_floors:
                for bb in bboxes_on_floors[int(f)]:
                    bbox_idx, _, bbox_coords = bb
                    img = self.paint_bbox(img, bbox_coords, self.args.colors[bbox_idx % 6]) 
            
            img = img.resize((self.image_size[2], self.image_size[1]))
            if "train" in self.mode:
                all_maps[enum, :, :, :] = self.preprocess_data_aug(img)[:3, :, :]
            else:
                all_maps[enum, :, :, :] = self.preprocess(img)[:3, :, :]
            all_conversions[enum, :] = self.mesh2meters[sn][f]["threeMeterRadius"] / 3.0


        return all_maps, all_conversions

    def gather_true_floor(self, index, paint_box):
        f = self.levels[index]
        sn = self.scan_names[index]

        img = Image.open(
                "{}floor_{}/{}_{}.png".format(self.args.image_dir, f, sn, f)
            )
        if paint_box:
            for bb in bboxes_on_floors[int(f)]:
                    bbox_idx, _, bbox_coords = bb
                    img = self.paint_bbox(img, bbox_coords, self.args.colors[bbox_idx % 6]) 

        img = img.resize((self.image_size[2], self.image_size[1]))

        true_map = self.preprocess_data_aug(img)[:3, :, :]
        true_conversion = self.mesh2meters[sn][f]["threeMeterRadius"] / 3.0

        return true_map, true_conversion

        

    def paint_bbox(self, image, bbox, color):
        x1, y1, x2, y2 = bbox
        overlay = Image.new('RGBA', image.size, color+(0,))
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        draw.rectangle((x1, y1, x2, y2), fill=color+(50,))

        # Alpha composite these two images together to obtain the desired result.
        image = Image.alpha_composite(image, overlay)
        return image 

# Old Room level segmentation method 
    # def gather_all_floors(self, index):
    #     all_maps = torch.zeros(
    #         self.args.max_floors,
    #         self.image_size[0],
    #         self.image_size[1],
    #         self.image_size[2],
    #     )
    #     region_segments = torch.zeros(
    #         self.args.max_floors,
    #         1,
    #         self.args.ds_height,
    #         self.args.ds_width,
    #     )
    #     all_conversions = torch.zeros(self.args.max_floors, 1)
    #     sn = self.scan_names[index]
    #     floors = self.mesh2meters[sn].keys()
    #     base = 0
    #     for enum, f in enumerate(floors):
    #         img = Image.open(
    #             "{}floor_{}/{}_{}.png".format(self.args.image_dir, f, sn, f)
    #         )
            
    #         # Get region annotations for the image corresponding to floor f of scan sn
    #         with open("{}floor_{}/{}_{}.json".format(self.args.image_dir, f, sn, f)) as file:
    #             annotationDict = json.load(file)

    #         # Assign the correct segment label to the 2D image of the floor
    #         segment_data = self.annotateImageWithSegmentationData(img, annotationDict, base)
    #         segment_data = cv2.resize(segment_data, (self.args.ds_width, self.args.ds_height), interpolation = cv2.INTER_NEAREST)
    #         region_segments[enum, :, :, :] = torch.as_tensor(segment_data)
            
    #         img = img.resize((self.image_size[2], self.image_size[1]))
    #         if "train" in self.mode:
    #             all_maps[enum, :, :, :] = self.preprocess_data_aug(img)[:3, :, :]
    #         else:
    #             all_maps[enum, :, :, :] = self.preprocess(img)[:3, :, :]
    #         all_conversions[enum, :] = self.mesh2meters[sn][f]["threeMeterRadius"] / 3.0
    #         base = int(np.max(segment_data))
    #     region_segments = region_segments.long()
    #     return all_maps, all_conversions, region_segments
    
    # def mask_maps(self, index, maps, target, room):
    #     f = int(self.levels[index])
    #     sn = self.scan_names[index]

    #     mask = torch.zeros(
    #         self.args.max_floors,
    #         self.image_size[0],
    #         self.image_size[1],
    #         self.image_size[2],
    #     )

    #     with open("{}floor_{}/{}_{}.png".format(self.args.image_dir, f, sn, f)) as f:
    #         annotation_dict = json.load(f)
    #     p = annotation_dict['shapes'][room]
    #     np_polygon = np.array(p['points']).T
    #     cc, rr = polygon(np_polygon[0], np_polygon[1])
    #     mask[f, rr, cc] = 1

    #     # masked_image = torch.where(mask == 1, maps, 0)
    #     masked_image = maps[ mask == 1]

    #     return masked_image


    


    def get_info(self, index):
        info_elem = [
            self.dialogs[index],
            self.levels[index],
            self.scan_names[index],
            self.annotation_ids[index],
            self.viewPoint_location[index],
        ]
        return info_elem
    
    def gather_all_rooms(self, index):
        all_rooms = torch.zeros((36, 3, 224, 224))
        scan_name = self.scan_names[index]
        

        for room in os.listdir(f'../../data/rooms/{scan_name}'):
            label = room.split('.')[0]
            image = Image.open(f'../..//data/rooms/{scan_name}/{room}')
            image_tensor = self.args.clip_preprocess(image)[:3, :, :]
            all_rooms[int(label), :, :, :] = image_tensor
        
        return all_rooms
    
    # def gather_all_floors(self, index):
    #     all_maps = torch.zeros(
    #         self.args.max_floors,
    #         self.image_size[0],
    #         self.image_size[1],
    #         self.image_size[2],
    #     )
    #     all_conversions = torch.zeros(self.args.max_floors, 1)
    #     sn = self.scan_names[index]
    #     floors = self.mesh2meters[sn].keys()
    #     base = 0
    #     for enum, f in enumerate(floors):
    #         img = Image.open(
    #             "{}floor_{}/{}_{}.png".format(self.args.image_dir, f, sn, f)
    #         )
            

    #         img = img.resize((self.image_size[2], self.image_size[1]))
    #         if "train" in self.mode:
    #             all_maps[enum, :3, :, :] = self.preprocess_data_aug(img)[:3, :, :]
    #         else:
    #             all_maps[enum, :3, :, :] = self.preprocess(img)[:3, :, :]
    #         all_maps[enum, 3, :, :] = torch.as_tensor(segmentData)
    #         all_conversions[enum, :] = self.mesh2meters[sn][f]["threeMeterRadius"] / 3.0
    #         base = int(np.max(segmentData))
    #     segmentDataExtracted = all_maps[:, -1, :, :]  # all_maps[:, -1, :, :] returns the mask containing segmentation information. Last channel contains ids of the region segments
    #     return all_maps, all_conversions, segmentDataExtracted

    def annotateImageWithSegmentationData(self, image, annotationDict, base):
        image = np.array(image)
        img_shape = image.shape
        grandMask = np.zeros((img_shape[0], img_shape[1]))
        for p in annotationDict['shapes']:
            np_polygon = np.array(p['points']).T
            cc, rr = polygon(np_polygon[0], np_polygon[1]) # This is inverted because row and column are stores as x, y co-ordinates in the polygon data
            grandMask[rr, cc] = base + int(p['label'])
        return grandMask 
    
    def create_similarity_target(self, index, location):
        floor = int(self.levels[index])
        row = location[0] // self.region_size[1]
        col = location[1] // self.region_size[2]
        target = torch.zeros(30)
        target[6*floor + 3*row + col] = 1 
        target[target == 0] = -1 

        return target 


    def create_target(self, index, location, mesh_conversion):
        gaussian_target = np.zeros(
            (self.args.max_floors, self.image_size[1], self.image_size[2])
        )
        gaussian_target[int(self.levels[index]), location[0], location[1]] = 1
        gaussian_target[int(self.levels[index]), :, :] = gaussian_filter(
            gaussian_target[int(self.levels[index]), :, :],
            sigma=mesh_conversion,
        )
        gaussian_target[int(self.levels[index]), :, :] = (
            gaussian_target[int(self.levels[index]), :, :]
            / gaussian_target[int(self.levels[index]), :, :].sum()
        )
        gaussian_target = torch.tensor(gaussian_target)
        gaussian_target = (
            nn.functional.interpolate(
                gaussian_target.unsqueeze(1),
                (self.args.ds_height, self.args.ds_width),
                mode="bilinear",
            )
            .squeeze(1)
            .float()
        )
        gaussian_target = gaussian_target / gaussian_target.sum()
        return gaussian_target

    #  def create_classification_target(self, index, location, segmentData):
    #     return segmentData[int(self.levels[index]), location[0], location[1]]

    def __getitem__(self, index):
        location = copy.deepcopy(self.locations[index])

        # x->0.56, y->0.64

        # region_location = copy.deepcopy(self.locations[index])

        location = np.round(np.asarray(location) * self.args.ds_percent).astype(int)
        
        mesh_conversion = self.mesh_conversions[index] * self.args.ds_percent
        # mesh_conversion = [ self.mesh_conversions[index] * 0.64, self.mesh_conversions[index] *0.56  ]
        # text = torch.LongTensor(self.texts[index])
        text = clip.tokenize(self.dialogs[index], truncate=True)  
        text = text.squeeze(0) 
        # seq_length = np.array(self.seq_lengths[index])
        maps, conversions = self.gather_all_floors(index)
        # true_map, true_converion = self.gather_true_floor(index, False)
        # region_segments = region_segments.squeeze(1)
        # rooms = self.gather_all_rooms(index)
        # if self.mode == 'train':

        target = self.create_target(index, location, mesh_conversion)
        # target = torch.nn.functional.interpolate(
        #         target.unsqueeze(1),
        #         (self.args.ds_height, self.args.ds_width),
        #         mode="bilinear",
        #     ).squeeze(1).float()
        # target /= target.sum()

        # region_location[0] = int(region_location[0] * 0.64)
        # region_location[1] = int(region_location[1] * 0.56)

        # similarity_target = self.create_similarity_target(index, region_location)
        info_elem = self.get_info(index)
        return (
            info_elem,
            text,
            # seq_length,
            target,
            # similarity_target,
            maps,
            conversions,
        )

    def __len__(self):
        return len(self.annotation_ids)
