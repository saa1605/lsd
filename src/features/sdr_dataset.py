import torch 
from PIL import Image, ImageDraw
import json 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn as nn
import clip 

class SDRDataset(Dataset):
    def __init__(
        self,
        mode,
        args,
        texts,
        centers,
        panoids,
        route_ids,
    ):
        self.mode = mode
        self.args = args
        self.centers = centers
        self.texts = texts
        self.panoids = panoids
        self.route_ids = route_ids
        # if self.mode == 'train':
            # self.mode = 'train_augmented'
        # self.bbox_data = torch.load(f'{args.processed_save_path}/{args.rpn_mode}/best_bbox_arr_{self.mode}_{args.text_feature_mode}_all_floors.pt')
        self.image_size = [
            3,
            200,
            500,
        ]

        self.preprocess_data_aug = transforms.Compose(
            [
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
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], # [0.485, 0.456, 0.406, 0.555],
                    std= [0.26862954, 0.26130258, 0.27577711]# [0.229, 0.224, 0.225, 0.222],
                ),
            ]
        )

    def deconstruct_panoramas(self, index):
        all_maps = torch.zeros(
            self.args.max_floors,
            self.image_size[0],
            self.image_size[1],
            self.image_size[2],
        )
        panoid = self.panoids[index]
        # bboxes_on_floors = self.bbox_data[index]['floors']
        

        img = Image.open(
            f'{self.args.image_dir}/{panoid}.jpg'
        )
        for enum in range(self.args.max_floors):
            crop = img.crop((600*enum, 0, 600*(enum+1), 1500))
            crop = crop.resize((self.image_size[2], self.image_size[1]))
        # if int(f) in bboxes_on_floors:
        #     for bb in bboxes_on_floors[int(f)]:
        #         bbox_idx, _, bbox_coords = bb
        #         img = self.paint_bbox(img, bbox_coords, self.args.colors[bbox_idx % 6]) 
            print(all_maps.size(), self.preprocess_data_aug(crop).size())
            if "train" in self.mode:
                all_maps[enum, :, :, :] = self.preprocess_data_aug(crop)[:3, :, :]
            else:
                all_maps[enum, :, :, :] = self.preprocess(crop)[:3, :, :]


        return all_maps


    def paint_bbox(self, image, bbox, color):
        x1, y1, x2, y2 = bbox
        overlay = Image.new('RGBA', image.size, color+(0,))
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        draw.rectangle((x1, y1, x2, y2), fill=color+(50,))

        # Alpha composite these two images together to obtain the desired result.
        image = Image.alpha_composite(image, overlay)
        return image

    def create_target(self, index, level, location):

        gaussian_target = np.zeros(
            (self.args.max_floors, self.image_size[1], self.image_size[2])
        )
        gaussian_target[int(level), location[0], location[1]] = 1
        gaussian_target[int(level), :, :] = gaussian_filter(
            gaussian_target[int(level), :, :],
            sigma=40.,
        )
        gaussian_target[int(level), :, :] = (
            gaussian_target[int(level), :, :]
            / gaussian_target[int(level), :, :].sum()
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

    def get_info(self, index):
        info_elem = [
            self.texts[index],
            self.panoids[index],
            self.route_ids[index],
        ]
        return info_elem

    def __getitem__(self, index):
        print(self.centers[index])
        location = [self.centers[index]['y'], self.centers[index]['x']]
        level = self.centers[index]["x"]*3000  // (3000 // 5)
        location = np.round(np.asarray(location) * self.args.ds_percent).astype(int)
        text = clip.tokenize(self.texts[index], truncate=True)  
        text = text.squeeze(0) 
        maps = self.deconstruct_panoramas(index)
        target = self.create_target(index, level, location)
        info_elem = self.get_info(index)
        return (
            info_elem,
            text,
            target,
            maps,
        )

    def __len__(self):
        return len(self.annotation_ids)