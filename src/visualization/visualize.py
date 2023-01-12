import json 
import random 
from PIL import Image, ImageDraw
import torch 
import clip 
from ml_collections import config_dict
setup_dict = {
    # paths
    'raw_data_path': '/data1/saaket/lsd_data/data/raw',
    'interim_save_path': '/data1/saaket/lsd_data/data/interim',
    'processed_save_path': '/data1/saaket/lsd_data/data/processed',
    'valUnseen_data_file': '/data1/saaket/lsd_data/data/raw/way_splits/valUnseen_data.json',
    'train_data_file': '/data1/saaket/lsd_data/data/raw/way_splits/train_data.json',
    'valSenn_data_file': '/data1/saaket/lsd_data/data/raw/way_splits/valSeen_data.json',
    'scan_levels_file': '/data1/saaket/lsd_data/data/raw/floorplans/scan_levels.json',
    'node2pix_file': '/data1/saaket/lsd_data/data/raw/floorplans/allScans_Node2pix.json',
    'geodistance_file': '/data1/saaket/lsd_data/data/raw/geodistance_nodes.json',
    'mesh2meters_file': '/data1/saaket/lsd_data/data/raw/floorplans/pix2meshDistance.json',
    'floorplans_dir': '/data1/saaket/lsd_data/data/raw/floorplans',
    'figures_path': '../reports/figures',

    # model details 
    'clip_version': 'ViT/B32',
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu', 
    'tokenizer': clip.tokenize,
    # feature extraction modes 
    'data_mode': 'train',
    'text_feature_mode': 'one_utterance', 
    'rpn_mode': 'conventional',
    'colors': [(240,0,30), (155,50,210), (255,255,25), (0,10,255), (255,170,230), (0,255,0)],
    'color_names': ['red', 'purple', 'yellow', 'blue', 'pink', 'green']

}

config = config_dict.ConfigDict(setup_dict)

# To fix: Paths 
def visualize_bboxes(best_bbox_arr, example_num, idx):
    bbox_dict = best_bbox_arr[example_num]
    with open('../input/way-led/way_splits/valUnseen_data.json') as jj:
        valUnseen_data = json.load(jj)
    x, y = valUnseen_data[example_num]['finalLocation']['pixel_coord'] 
    ideal_floor = valUnseen_data[example_num]['finalLocation']['floor'] 
    
    floors = bbox_dict['floors']
    unique_floors = list(set(floors.keys()))
    try:
        viz_floor = unique_floors[idx]
    except:
        print("This floor does not include a good bounding box")
        return 
    image = Image.open(f'../input/way-led/floorplans/floor_{viz_floor}/{bbox_dict["scanName"]}_{viz_floor}.png')
    for box_num, utterance, bbox in floors[viz_floor]:
        x1, y1, x2, y2 = bbox
        draw = ImageDraw.Draw(image)
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        rgb = (r,g,b)
        draw.rectangle((x1, y1, x2, y2), width=2, outline=rgb)
        draw.text((x1, y1), utterance, fill=(r, g, b, 128))
    if viz_floor == ideal_floor:
        r = 10
        draw.ellipse([(x-r, y-r), (x+r, y+r)], fill = 'red', outline ='red')
        image.save('dump/led_example.png')
    return (image, bbox_dict['floors'][viz_floor], viz_floor,bbox_dict['scanName'])

if __name__=='__main__':
    best_bbox_arr = torch.load(f'{config.processed_save_path}/{config.rpn_mode}/best_bbox_arr_vision_transformer_valUnseen_one_utterance_all_floors.pt')
    visualize_bboxes()