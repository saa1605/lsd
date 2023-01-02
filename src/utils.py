import torch
import numpy as np
import json
import networkx as nx
import numpy as np
import math
import torch.nn as nn
import os
import spacy
nlp = spacy.load("en_core_web_sm")
import re 
# from shapely.geometry import Point, Polygon

stop_words = ['bear',
 'touchdown',
 'row',
 'part',
 'one',
 'end',
 'base',
 'luck',
 'step',
 'way',
 'above',
 'after',
 'around',
 'before',
 'beginning',
 'behind',
 'below',
 'beside',
 'side',
 'between',
 'bottom',
 'down',
 'end',
 'back',
 'front',
 'far',
 'finish',
 'front in',
 'inside',
 'left',
 'middle',
 'near',
 'next to',
 'off',
 'on',
 'out',
 'outside',
 'over',
 'right',
 'start',
 'through',
 'top',
 'under',
 'up',
 'upside down',
 'center',
 'side']



def evaluate(args, splitFile, run_name):
    split_name = splitFile.split("_")[0]
    distance_scores = []
    splitData = json.load(open(args.data_dir + splitFile))
    geodistance_nodes = json.load(open(args.geodistance_file))
    fileName = f"{run_name}_{split_name}_submission.json"
    fileName = os.path.join(args.predictions_dir, fileName)
    submission = json.load(open(fileName))
    for gt in splitData:
        gt_vp = gt["finalLocation"]["viewPoint"]
        pred_vp = submission[gt["episodeId"]]["viewpoint"]
        dist = geodistance_nodes[gt["scanName"]][gt_vp][pred_vp]
        distance_scores.append(dist)

    distance_scores = np.asarray(distance_scores)
    print(
        f"Result {split_name} -- \n LE: {np.mean(distance_scores):.4f}",
        f"Acc@0m: {sum(distance_scores <= 0) * 1.0 / len(distance_scores):.4f}",
        f"Acc@3m: {sum(distance_scores <= 3) * 1.0 / len(distance_scores):.4f}",
        f"Acc@5m: {sum(distance_scores <= 5) * 1.0 / len(distance_scores):.4f}",
        f"Acc@10m: {sum(distance_scores <= 10) * 1.0 / len(distance_scores):.4f}",
    )

def euclidean_distance_from_pixels(args, preds, mesh_conversions, info_elem, mode, target_coords):
    node2pix = json.load(open(args.image_dir + "allScans_Node2pix.json"))
    geodistance_nodes = json.load(open(args.geodistance_file))
    distances, episode_predictions = [], []
    dialogs, levels, scan_names, episode_ids, true_viewpoints = info_elem
    for pred, conversion, sn, tv, id in zip(
        preds, mesh_conversions, scan_names, true_viewpoints, episode_ids
    ):
        total_floors = len(set([v[2] for k, v in node2pix[sn].items()]))
        pred = nn.functional.interpolate(
            pred.unsqueeze(1), (700, 1200), mode="bilinear"
        ).squeeze(1)[:total_floors]
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        # convers = conversion.view(args.max_floors, 1, 1)[pred_coord[0].item()]
        convers = conversion[true_level]
        if pred_coord[0] == true_level:
            eu_dist = np.sqrt((pred_coord[1] - location[1]) ** 2 + (pred_coord[2] - location[0]) ** 2) // convers
            distances.append(eu_dist) 
        else:
            distances.append(11)

        
        




def accuracy(dists, threshold=3):
    """Calculating accuracy at 3 meters by default"""
    return np.mean((torch.tensor(dists) <= threshold).int().numpy())


def accuracy_batch(dists, threshold):
    return (dists <= threshold).int().numpy().tolist()


def distance(pose1, pose2):
    """Euclidean distance between two graph poses"""
    return (
        (pose1["pose"][3] - pose2["pose"][3]) ** 2
        + (pose1["pose"][7] - pose2["pose"][7]) ** 2
        + (pose1["pose"][11] - pose2["pose"][11]) ** 2
    ) ** 0.5


def open_graph(connectDir, scan_id):
    """Build a graph from a connectivity json file"""
    infile = "%s%s_connectivity.json" % (connectDir, scan_id)
    G = nx.Graph()
    with open(infile) as f:
        data = json.load(f)
        for i, item in enumerate(data):
            if item["included"]:
                for j, conn in enumerate(item["unobstructed"]):
                    if conn and data[j]["included"]:
                        assert data[j]["unobstructed"][i], "Graph should be undirected"
                        G.add_edge(
                            item["image_id"],
                            data[j]["image_id"],
                            weight=distance(item, data[j]),
                        )
    return G


def get_geo_dist(D, n1, n2):
    return nx.dijkstra_path_length(D, n1, n2)


def snap_to_grid(geodistance_nodes, node2pix, sn, pred_coord, conversion, level):
    min_dist = math.inf
    best_node = ""
    for node in node2pix[sn].keys():
        if node2pix[sn][node][2] != int(level) or node not in geodistance_nodes:
            continue
        target_coord = [node2pix[sn][node][0][1], node2pix[sn][node][0][0]]
        dist = np.sqrt(
            (target_coord[0] - pred_coord[0]) ** 2
            + (target_coord[1] - pred_coord[1]) ** 2
        ) / (conversion)
        if dist.item() < min_dist:
            best_node = node
            min_dist = dist.item()
    return best_node

def region_accuracy(preds, region_nums):
    predicted_regions = preds.argmax(dim=1)
    correct_count = (predicted_regions == region_nums).sum().item()

    return correct_count / len(predicted_regions)

def distance_from_pixels_regions(args, preds, mesh_conversions, info_elem, mode):
    node2pix = json.load(open(args.image_dir + "allScans_Node2pix.json"))
    geodistance_nodes = json.load(open(args.geodistance_file))
    distances, episode_predictions = [], []
    dialogs, levels, scan_names, episode_ids, true_viewpoints = info_elem
    for pred, conversion, sn, tv, id in zip(
        preds, mesh_conversions, scan_names, true_viewpoints, episode_ids
    ):
        total_floors = len(set([v[2] for k, v in node2pix[sn].items()])) * 8
        pred = nn.functional.interpolate(
            pred.unsqueeze(1), (350, 300), mode="bilinear"
        ).squeeze(1)[:total_floors]
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        this_floor = pred_coord[0].item() // 8 
        remainder = pred_coord[0].item() % 8
        this_row = (remainder // 4)  
        this_col = (remainder % 4)
        row = this_row * 350 + pred_coord[1].item() 
        col = this_col * 300 + pred_coord[2].item() 

        coord = (this_floor, row, col)

        convers = conversion.view(args.max_floors, 1, 1)[coord[0]]
        pred_viewpoint = snap_to_grid(
            geodistance_nodes[sn],
            node2pix,
            sn,
            [coord[1], coord[2]],
            convers,
            coord[0],
        )
        if mode != "test":
            dist = geodistance_nodes[sn][tv][pred_viewpoint]
            distances.append(dist)
        episode_predictions.append([id, pred_viewpoint])
    return distances, episode_predictions



def distance_from_pixels(args, preds, mesh_conversions, info_elem, mode):
    """Calculate distances between model predictions and targets within a batch.
    Takes the propablity map over the pixels and returns the geodesic distance"""
    node2pix = json.load(open(args.image_dir + "allScans_Node2pix.json"))
    geodistance_nodes = json.load(open(args.geodistance_file))
    distances, episode_predictions = [], []
    dialogs, levels, scan_names, episode_ids, true_viewpoints = info_elem
    for pred, conversion, sn, tv, id in zip(
        preds, mesh_conversions, scan_names, true_viewpoints, episode_ids
    ):
        total_floors = len(set([v[2] for k, v in node2pix[sn].items()]))
        pred = nn.functional.interpolate(
            pred.unsqueeze(1), (700, 1200), mode="bilinear"
        ).squeeze(1)[:total_floors]
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        convers = conversion.view(args.max_floors, 1, 1)[pred_coord[0].item()]
        pred_viewpoint = snap_to_grid(
            geodistance_nodes[sn],
            node2pix,
            sn,
            [pred_coord[1].item(), pred_coord[2].item()],
            convers,
            pred_coord[0].item(),
        )
        if mode != "test":
            dist = geodistance_nodes[sn][tv][pred_viewpoint]
            distances.append(dist)
        episode_predictions.append([id, pred_viewpoint])
    return distances, episode_predictions

def floor_accuracy(args, preds, target):
    """Calculate distances between model predictions and targets within a batch.
    Takes the propablity map over the pixels and returns the geodesic distance"""
    correct_count = 0 
    preds = preds.cpu()
    target = target.cpu()
    for i in range(len(preds)):
        pred_floor = np.unravel_index(preds[i].argmax(), preds[i].size())[0]
        target_floor = np.unravel_index(target[i].argmax(), target[i].size())[0]
        # print(pred_floor, target_floor)
        if pred_floor == target_floor:
            correct_count += 1 
    
    return correct_count / len(preds) 

def annotateImageWithSegmentationData(image, annotationDict):
    image = np.array(image)
    print(image.shape)
    segmentChannel = np.zeros(image.shape)
    imageShape = image.shape

    for r, c in image:
        point = Point(r, c)
        for segment in annotationDict:
            polygon = Polygon(segment['shapes']['points'])
            if polygon.contains(point):
                segmentChannel[r, c] = segment['shapes']['label']
                break 
    return segmentChannel


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_metrics(y_pred, y_true):
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    true_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    false_positives = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    false_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    accuracy = (true_positives + true_negatives) / len(y_true)
    assert not np.isnan(true_negatives)
    assert not np.isnan(false_positives)
    assert not np.isnan(false_negatives)
    assert not np.isnan(true_positives)

    if true_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        iou = true_positives / (true_positives + false_positives + false_negatives)
    else:
        precision = recall = f1_score = iou = float('NaN')
    return {
        "accuracy": accuracy,
        "TN": true_negatives,
        "FP": false_positives,
        "FN": false_negatives,
        "TP": true_positives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "true_mean": np.mean(y_true),
        "pred_mean": np.mean(y_pred),
    }




def distance_metric_sdr(preds, targets):
    """Calculate distances between model predictions and targets within a batch."""
    preds = preds.cpu()
    targets = targets.cpu()
    distances = []
    for pred, target in zip(preds, targets):
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        target_coord = np.unravel_index(target.argmax(), target.size())
        dist = np.sqrt((target_coord[0] - pred_coord[0]) ** 2 + (target_coord[1] - pred_coord[1]) ** 2)
        dist = dist * 8 
        distances.append(dist)
    return distances

# def distance_metric_sdr(preds, x_target, y_target):
#     """Calculate distances between model predictions and targets within a batch."""
#     preds = preds.cpu()
#     targets = targets.cpu()
#     distances = []
#     for pred, target in zip(preds, targets):
#         pred = nn.functional.interpolate(
#             pred.unsqueeze(1), (1500, 600), mode="bilinear"
#         ).squeeze(1)
#         # target = nn.functional.interpolate(
#         #     target.unsqueeze(1), (1500, 600), mode="bilinear"
#         # ).squeeze(1)
#         pred_coord = np.unravel_index(pred.argmax(), pred.size())
#         print(pred_coord)
        
#         x = pred_coord[0]*600 + pred_coord[2]
#         y = pred_coord[1]
#         dist = np.sqrt((x_target - x) ** 2 + (y_target - y) ** 2)
#         distances.append(dist)
#     return distances

def accuracy_sdr(distances, margin=10):
    """Calculating accuracy at 80 pixel by default"""
    num_correct = 0
    for distance in distances:
        num_correct = num_correct + 1 if distance < margin else num_correct
    return num_correct / len(distances)



def removearticles(text):
    return  re.sub('^the |a |an |The |A |An ', '', text).strip()

def find_objects_in_text(text):
    doc = nlp(text)
    objects_in_image = []
    for oid, obj in enumerate(doc.noun_chunks):
        tags = [t.tag_ for t in nlp(obj.text)]
        start = obj.start
        end = obj.end
        if 'NN' in tags or 'NNS' in tags:
            contains_stop_word = False
            num_nouns = 0 
            for word in nlp(obj.text):
                if word.pos_ == 'NOUN':
                    num_nouns += 1 
            for word in nlp(obj.text):
                if word.pos_ == 'NOUN' and word.text.lower() in stop_words and num_nouns <= 1:
                    contains_stop_word = True
                    break 
            if not contains_stop_word:
                objects_in_image.append([removearticles(obj.text), start, end])
    if len(objects_in_image) < 1:
        objects_in_image.append((text, 0, len(text)))
    return objects_in_image
    
def add_prompt_to_touchdown_text(text, object_in_text, colors):
    text2color = {}
    doc = nlp(text)
    doc_list = list(doc)
    doc_list = [d.text for d in doc_list]
    color_counter = 0
    color_order = []
    duplicates = []
    for oid, obj in enumerate(list(reversed(object_in_text))):
        obj_str, start, end = obj 
        if obj_str not in text2color:
            text2color[obj_str] = colors[color_counter % len(colors)]
            
            duplicates.append(0)
        else:
            duplicates.append(1)
        color_order.append(text2color[obj_str])
        doc_list.insert(end, f'in {text2color[obj_str]} color')
        color_counter += 1
    out = " ".join(doc_list)
    return re.sub(r'\s([?.!"](?:\s|$))', r'\1', out.strip()), list(reversed(color_order)), duplicates 


        