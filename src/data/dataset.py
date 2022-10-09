from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import torch 
from typing import List, Tuple
from torch import Tensor
import time 

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

class AnchorImageDatasetSDR(Dataset):
    def __init__(self, image_list, transforms, coords):
        self.image_list = image_list 
        self.pano_slice_path = '/data1/saaket/lsd_data/data/processed/pano_slices' 
        self.arranged_image_list = self.arrange_images()
        self.transforms = transforms
        self.coords = coords # 1000 coords per image 
    
    def __len__(self):
        return len(self.arranged_image_list)

    def arrange_images(self):
        arranged_image_list = []
        for parent_image in self.image_list:
            child_images = [f'{self.pano_slice_path}/{parent_image}/{parent_image}_{i}.jpg' for i in range(8)]
            arranged_image_list.extend(child_images)
        return arranged_image_list
    
    def normalize_boxes(self, coord):
        if coord[1] > 799:
            coord[1] = 799
        if coord[3] > 799:
            coord[3] = 799
        if coord[0] > 459:
            coord[0] = 459 
        if coord[2] > 459:
            coord[2] = 459
        return coord

    def __getitem__(self, idx):
        img = Image.open(self.arranged_image_list[idx])
        anchor_list = []
        coord = self.coords[idx][:200]

        # cord_tfms_start = time.time()
        for i, c in enumerate(coord):
            c = self.normalize_boxes(c)
            anchor_list.append(self.transforms(img.crop(c)))
        # cord_tfms_end = time.time()
        # print('coord crop + tfms time: ', cord_tfms_end - cord_tfms_start)

        return torch.stack(anchor_list)

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
        self.pano_slice_path = '/data1/saaket/lsd_data/data/processed/pano_slices' 
        self.arranged_image_list = self.arrange_images()
        self.transforms = transforms
    
    def __len__(self):
        return len(self.arranged_image_list)

    def arrange_images(self):
        arranged_image_list = []
        for parent_image in self.image_list:
            child_images = [f'{self.pano_slice_path}/{parent_image}/{parent_image}_{i}.jpg' for i in range(8)]
            arranged_image_list.extend(child_images)
        return arranged_image_list

    def __getitem__(self, idx):
        img = Image.open(self.arranged_image_list[idx])
        try:
            img_tfms = self.transforms(img)
        except:
            print(self.arranged_image_list[idx] + ' has failed')
        return img_tfms 