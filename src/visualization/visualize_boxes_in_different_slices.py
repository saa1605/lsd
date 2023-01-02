import torch 
from tqdm import tqdm
from PIL import Image, ImageDraw
import json 

def touchdown_loader(data_path):
    texts, panoids = [], []
    with open(data_path) as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            panoids.append(obj['main_pano'])
            texts.append(obj['td_location_text'])
    return texts, panoids 

def draw_boxes_in_different_slices(bboxes, panoid):
    for box in bboxes:
        slice1, x1, y1, slice2, x2, y2 = box   
        if slice1 != slice2:
            for slice in range(slice1, slice2 + 1):
                if slice == slice1:
                    image = Image.open(f'/data1/saaket/lsd_data/data/processed/pano_slices/{panoid}/{panoid}_{slice}.jpg')   
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((x1, y1, image.width, y2))    
                elif slice == slice2:
                    image = Image.open(f'/data1/saaket/lsd_data/data/processed/pano_slices/{panoid}/{panoid}_{slice}.jpg')   
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, y1, x2, y2)) 
                else:
                    image = Image.open(f'/data1/saaket/lsd_data/data/processed/pano_slices/{panoid}/{panoid}_{slice}.jpg')   
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, y1, image.width, y2)) 
                image.save(f'dump/{panoid}_{slice}.jpg')
        else:
            image = Image.open(f'/data1/saaket/lsd_data/data/processed/pano_slices/{panoid}/{panoid}_{slice1}.jpg')   
            draw = ImageDraw.Draw(image)
            draw.rectangle((x1, y1, x2, y2))
            image.save(f'dump/{panoid}_{slice1}_unique.jpg')




def main():
    bounding_boxes = torch.load('/data2/saaket/mapped_best_train_bboxes_ViT_full_pano.pth') 
    texts, pano_names = touchdown_loader('/data1/saaket/touchdown/data/train.json')
    
    for idx, pano_name in enumerate(pano_names[:10]):
       draw_boxes_in_different_slices(bounding_boxes[idx], pano_name)
       print(texts[idx], pano_names[idx])

if __name__=='__main__':
    main()