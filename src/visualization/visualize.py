import json 
import random 
from PIL import Image, ImageDraw

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
    for utterance, bbox in floors[viz_floor]:
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
    return (image, bbox_dict['floors'][viz_floor], viz_floor,bbox_dict['scanName'])