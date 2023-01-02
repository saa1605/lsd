import os
import json 
from tqdm import tqdm 
import re 
from pathlib import Path
import numpy as np 
import spacy

nlp = spacy.load("en_core_web_sm")




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
    train_perspective_feature_files = [
        [
            '/data2/saaket/features_image/image_features_sdr_perspective_train0_10.pth',
            '/data2/saaket/features_image/image_features_sdr_perspective_train10_20.pth',
            '/data2/saaket/features_image/image_features_sdr_perspective_train20_30.pth',
            '/data2/saaket/features_image/image_features_sdr_perspective_train30_40.pth',
            '/data2/saaket/features_image/image_features_sdr_perspective_train40_5000.pth',
            '/data2/saaket/features_image/image_features_sdr_perspective_train5000_10000.pth',
        ],
        [
            '/data2/saaket/features_image/image_features_sdr_perspective_train10000_15000.pth',
            '/data2/saaket/features_image/image_features_sdr_perspective_train15000_20000.pth',
        ],
        [
            '/data2/saaket/features_image/image_features_sdr_perspective_train20000_25000.pth',
            '/data2/saaket/features_image/image_features_sdr_perspective_train25000_30000.pth',
        ],
        [
            '/data2/saaket/features_image/image_features_sdr_perspective_train30000_35000.pth',
            
        ],
        [
            '/data2/saaket/features_image/image_features_sdr_perspective_train35000_44456.pth',
        ]
    ]
dev_perspective_feature_files = [
    '/data2/saaket/features_image/image_features_sdr_perspective_dev_5000.pth',
    '/data2/saaket/features_image/image_features_sdr_perspective_dev_end.pth',
]


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


def main():
    mode = 'train'
    unique_panos = get_unique_panos(paths.train)
    texts, panoids = touchdown_loader(paths.train)
    pano2id, id2pano = get_pano2id(unique_panos)
    unique_pano_slices = []
    for pano in unique_panos:
        for i in range(8):
            unique_pano_slices.append(f'{pano}_{i}')
    # Pano Dicts 
    pano2id, id2pano = get_pano2id(unique_panos)
    slice2id, id2slice = get_pano2id(unique_pano_slices)

    object_in_text_arr = []

    for i in tqdm(range(len(panoids))):
       object_in_text_arr.append(find_objects_in_text(texts[i]))
    
    np.save(f'/data2/saaket/objects_in_text_{mode}.npy', object_in_text_arr)

if __name__ == '__main__':
    main()