#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:49:51 2018
prepare the mscoco data into h5 file format
@author: danny
"""

import os
import json
import pickle
from visual_features import vis_feats
from text_features import text_features_coco
from collections import defaultdict
import tables
# paths to the coco caption and image files 
train_img_path = os.path.join('/data/mscoco/train2017')
val_img_path = os.path.join('/data/mscoco/val2017')
text_path = os.path.join('/data/mscoco/annotations')
# save the resulting feature file here
data_loc = os.path.join('/prep_data/coco_features.h5')

def batcher(batch_size, dictionary):
    keys = [x for x in dictionary]
    for start_idx in range(0, len(dictionary) - 1, batch_size):
        excerpt = {}
        if not start_idx + batch_size > len(dictionary):
            for x in range(start_idx, start_idx + batch_size):
                excerpt[keys[x]] = dictionary[keys[x]]
            yield excerpt
        else:
            for x in range(start_idx, len (dictionary)):
                excerpt[keys[x]] = dictionary[keys[x]]
            yield excerpt

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
    
# list the image and annotations directory
train_imgs = os.listdir(train_img_path)
val_imgs = os.listdir(val_img_path)

annotations = [os.path.join(text_path, x) for x in os.listdir(text_path)]
annotations.sort()
# load the validation annotations
val_cap = json.load(open(annotations[1]))
val_cap = val_cap['annotations']
# load the training annotations
train_cap = json.load(open(annotations[0]))
train_cap = train_cap['annotations']

# prepare dictionaries with all the captions and file ids
train_dict = defaultdict(list)  
for x in train_cap:
   key = str(x['image_id'])
   while len(key) < 6:
       key = '0' + key
   train_dict[key] = train_dict[key] + [x]
       
val_dict = defaultdict(list)   
for x in val_cap:
   key = str(x['image_id'])
   while len(key) < 6:
       key = '0' + key
   val_dict[key] = val_dict[key] + [x]

# strip the files to their basename (remove file extension)
train_imgs_base = [x.split('.')[0] for x in train_imgs]
val_imgs_base = [x.split('.')[0] for x in val_imgs]
# create a dictionary with image id pointing to the location of the image file
train_img = {}
for im in train_imgs_base:
    train_img[im.split('_')[-1][-6:]] = [im + '.jpg']
val_img = {}
for im in val_imgs_base:
    val_img[im.split('_')[-1][-6:]] = [im + '.jpg']

# create h5 output file for preprocessed images and audio
output_file = tables.open_file(data_loc, mode='a')

# we need to append something to the flickr files names because pytable group names cannot start
# with integers.
append_name = 'coco_'

# the size of this database is bigger than the maximum recommended amount of children
# a node can have so they cannot all be directly under the root node. instead the 
# database is split in smaller 10k bits.
count = 0
subgroups = []
for batch in batcher(10000, train_img):
    node = output_file.create_group('/', 'subgroup_' + str(count))
    subgroups.append(node)
    for x in batch:
        output_file.create_group(node, append_name + x.split('.')[0])
    count +=1
# list all the nodes containing training instances
train_node_list = []
for subgroup in subgroups:
    train_node_list = train_node_list + subgroup._f_list_nodes()

subgroups = []
for batch in batcher(10000, val_img):
    node = output_file.create_group('/', 'subgroup_' + str(count))
    subgroups.append(node)
    for x in batch:
        output_file.create_group(node, append_name + x.split('.')[0])
    count +=1     
# list all the nodes containing validation instances
val_node_list = []
for subgroup in subgroups:
    val_node_list = val_node_list + subgroup._f_list_nodes()
    
# create the visual features for all images  
vis_feats(val_img_path, output_file, append_name, val_img, val_node_list, 'resnet')
vis_feats(train_img_path, output_file, append_name, train_img, train_node_list, 'resnet') 

# add text features for all captions
text_features_coco(train_dict, output_file, append_name, train_node_list)
text_features_coco(val_dict, output_file, append_name, val_node_list)
# close the output files
output_file.close()
