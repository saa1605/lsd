import numpy as np 
import torch 
import json 
from ml_collections import config_dict
import torch 
import clip 

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
    'clip_version': 'RN50',
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
import pprint
pp = pprint.PrettyPrinter(indent=4)
import matplotlib.pyplot as plt
# To Fix: paths 
def inside_bbox(x, y, bbox):
    # Check if a point lies fully inside a bbox
    # bbox [x1, y1, x2, y2]
    # x increases to the right, y increases downwards
    x_left, y_top, x_right, y_bottom = bbox
    if x >= x_left and x <= x_right and y>=y_top and y<=y_bottom:
        return True
    else:
        return False


def closest_in_scan(cx, cy, bbox_floor, node2pix, scan_name):
    closest_vp = None
    closest_dist = float('inf')
    c_point = np.array([cx, cy])
    closest_point = 0, 0
    for viewpoint in node2pix[scan_name]:
        pixels, real, floor = node2pix[scan_name][viewpoint]
        if floor == bbox_floor:
            v_point = np.array(pixels)
            dist = np.linalg.norm(c_point - v_point)
            if dist <= closest_dist:
                closest_vp = viewpoint
                closest_dist = dist
                closest_point = tuple(pixels)
    return closest_vp, closest_point
        
    
def get_viewpoints_inside_bbox(bbox, bbox_floor, node2pix, scan_name):
    viewpoints = node2pix[scan_name]
    points_in_bbox = []
    coordinates_in_bbox = []
    for viewpoint in viewpoints:
        v_x, v_y = node2pix[scan_name][viewpoint][0]
        vp_floor = node2pix[scan_name][viewpoint][-1]
        if vp_floor == bbox_floor and inside_bbox(v_x, v_y, bbox):
            points_in_bbox.append(viewpoint)
            coordinates_in_bbox.append((v_x, v_y))
    if len(points_in_bbox) == 0:
        cx, cy = ( bbox[0] + bbox[2] ) // 2 , ( bbox[1] + bbox[3] ) // 2
        approx_viewpoint, approx_coords = closest_in_scan(cx, cy, bbox_floor, node2pix, scan_name)
        points_in_bbox.append(approx_viewpoint)
        coordinates_in_bbox.append(approx_coords)
    return points_in_bbox, coordinates_in_bbox
            
def get_closest_viewpoint_distance(viewpoints, gt_viewpoint, node2pix, geodistance_nodes, scan_name):
    closest_distance = float('inf')
    closest_vp = None
    for v in viewpoints:
        dist = geodistance_nodes[scan_name][gt_viewpoint][v]
        if dist < closest_distance:
            closest_distance = dist
            closest_vp = v
    return closest_distance, closest_vp  
    
def compute(best_bbox_arr, config):
    # 1. Average distance to closest viewpoint that is inside a bounding box or closest to the center of the bounding box if not point in bbox
    allscans_node2pix = json.load(open(config.node2pix_file))
    geodistance_nodes = json.load(open(config.geodistance_file))
    # Combined 
    empties = []
    dists = []
    failed = []

    for candidate in best_bbox_arr:
        floors = list(candidate['floors'].keys())
        view_points_in_all_bboxes = []
        for floor in floors:
            for item in candidate['floors'][floor]:
                box_floor, dialog, bbox = item
                view_points_in_bbox, _ = get_viewpoints_inside_bbox(bbox, floor, allscans_node2pix, candidate['scanName'])
                view_points_in_all_bboxes.extend(view_points_in_bbox)
        try:
            closest_dist, _ = get_closest_viewpoint_distance(view_points_in_all_bboxes, candidate['finalLocation']['viewPoint'], allscans_node2pix, geodistance_nodes, candidate['scanName'])
            dists.append(closest_dist)
        except:
            failed.append((view_points_in_all_bboxes, candidate['finalLocation']['viewPoint'], candidate['scanName']))
#         print('_____________________')
    return dists, failed



if __name__ == '__main__':
    best_bbox_arr = torch.load(f'{config.processed_save_path}/{config.rpn_mode}/best_bbox_arr_vision_transformer_valUnseen_one_utterance_all_floors.pt')
    dist, failed = compute(best_bbox_arr, config) 
    dist_np = np.array(dist)
    for i in [1, 3, 5, 10]:
        pp.pprint((f'bboxes close to {i}', (dist_np <= i).sum() / len(dist_np) ))
    print(len(dist_np))
    plt.hist(dist, bins = list(range(0, 30)))
    plt.savefig(f'{config.figures_path}/bbox_dist_distribution_{config.rpn_mode}_{config.text_feature_mode}_{config.data_mode}_vision_transformer.png')
        
    





