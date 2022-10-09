import torch 
from PIL import Image, ImageDraw
import cv2 
import json 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn as nn
import clip 
import time 

class SDRDataset(Dataset):
    def __init__(
        self,
        mode,
        args,
        texts,
        seq_lengths,
        centers,
        perspective_targets,
        panoids,
        route_ids,
    ):
        self.mode = mode
        self.args = args
        self.centers = centers
        self.perspective_targets = perspective_targets
        self.texts = texts
        self.seq_lengths = seq_lengths
        self.panoids = panoids
        self.route_ids = route_ids
        # if self.mode == 'train':
            # self.mode = 'train_augmented'
        # self.bbox_data = torch.load(f'{args.processed_save_path}/{args.rpn_mode}/best_bbox_arr_{self.mode}_{args.text_feature_mode}_all_floors.pt')
        self.image_size = [
            3,
            800,
            460,
        ]

        self.preprocess_data_aug = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.5, hue=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], # [0.485, 0.456, 0.406, 0.555],
                    std= [0.26862954, 0.26130258, 0.27577711]# [0.229, 0.224, 0.225, 0.222],
                ),
            ]
        )
        self.preprocess = transforms.Compose(
            [
                # transforms.ToPILImage(), 
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], # [0.485, 0.456, 0.406, 0.555],
                    std= [0.26862954, 0.26130258, 0.27577711]# [0.229, 0.224, 0.225, 0.222],
                ),
            ]
        )

    def deconstruct_panoramas(self, index):
        all_maps = torch.zeros(
            self.args.pano_slices,
            self.image_size[0],
            self.image_size[1],
            self.image_size[2],
        )
        panoid = self.panoids[index]
        # bboxes_on_floors = self.bbox_data[index]['floors']
        # img = cv2.imread(
        #     f'{self.args.image_dir}/{panoid}.jpg'
        # )
        h_stride = 45
        nn_heading_angles = list(range(-180, 180, h_stride))
        
        for enum, heading_angle in enumerate(nn_heading_angles):
            # if int(f) in bboxes_on_floors:
            #     for bb in bboxes_on_floors[int(f)]:
            #         bbox_idx, _, bbox_coords = bb
            #         img = self.paint_bbox(img, bbox_coords, self.args.colors[bbox_idx % 6]) 
            # perspective_image = self.get_perspective(img, FOV=60, THETA=heading_angle, PHI=0, height=800, width=460)
            perspective_image = Image.open(f'{self.args.image_dir}/{panoid}/{panoid}_{enum}.jpg')
            if "train" in self.mode:
                all_maps[enum, :, :, :] = self.preprocess_data_aug(perspective_image)[:3, :, :]
            else:
                all_maps[enum, :, :, :] = self.preprocess(perspective_image)[:3, :, :]
        return all_maps
    
    # Persepctive transforming the image 
    def get_perspective(self, img, FOV, THETA, PHI, height, width, RADIUS=128):
        img = np.array(img)
        img_height, img_width, _ = img.shape
        equ_h = img_height
        equ_w = img_width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([height, width, 3], np.float)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        persp = cv2.remap(img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_WRAP)
        
        return persp


    def paint_bbox(self, image, bbox, color):
        x1, y1, x2, y2 = bbox
        overlay = Image.new('RGBA', image.size, color+(0,))
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        draw.rectangle((x1, y1, x2, y2), fill=color+(50,))

        # Alpha composite these two images together to obtain the desired result.
        image = Image.alpha_composite(image, overlay)
        return image

    def create_target(self, index, mesh_conversion=20):
        gaussian_target = np.zeros(
            (self.args.pano_slices, self.image_size[1], self.image_size[2])
        )
        target_slice_number, target_x, target_y = self.perspective_targets[index]

        gaussian_target[target_slice_number, target_y, target_x] = 1
        gaussian_target = np.transpose(gaussian_target, (1, 0, 2))
        gaussian_target = np.expand_dims(np.reshape(gaussian_target, (self.image_size[1], self.args.pano_slices * self.image_size[2])), 0)
        gaussian_target[0, :, :] = gaussian_filter(
            gaussian_target[0, :, :],
            sigma=mesh_conversion,
        )
        gaussian_target[0, :, :]= (
            gaussian_target[0, :, :]
            / gaussian_target.sum()
        )
        gaussian_target = torch.tensor(gaussian_target)
        gaussian_target = (
            nn.functional.interpolate(
                gaussian_target.unsqueeze(1),
                (self.args.sdr_ds_height, self.args.sdr_ds_width),
                mode="bilinear",
            )
            .squeeze(1)
            .float()
        )
        gaussian_target = gaussian_target / gaussian_target.sum()
        gaussian_target = gaussian_target.squeeze(0)
        
        return gaussian_target

    def get_info(self, index):
        info_elem = [
            int(self.centers[index]['y'] * 1500),
            int(self.centers[index]['x'] * 3000),
            self.texts[index],
            self.panoids[index],
            self.route_ids[index],
        ]
        return info_elem

    def __getitem__(self, index):
        if self.args.model == 'clip':
            text = clip.tokenize(self.texts[index], truncate=True)  
            text = text.squeeze(0) 
        else:
            text = self.texts[index]
        # pano_decon_start = time.time()
        maps = self.deconstruct_panoramas(index)
        # pano_decon_end = time.time()
        # print("Pano Deconstruction Time: ", (pano_decon_end - pano_decon_start))
        # target_creation_start = time.time()
        target = self.create_target(index)
        # target_creation_end = time.time()
        # print("Target Construction Time: ", (target_creation_end - target_creation_start))
        info_elem = self.get_info(index)
        seq_length = np.array(self.seq_lengths[index])
        return {
            "info_elem": info_elem,
            "text": text,
            "seq_length": seq_length, 
            "target": target,
            "maps": maps,
        }

    def __len__(self):
        return len(self.route_ids)