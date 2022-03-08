#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:22:41 2021

@author: danny
"""
from collections import defaultdict
import json
import tables

def read_data(h5_file):       
    h5_file = tables.open_file(h5_file, 'r')
    for subgroup in h5_file.root:
        for node in subgroup:
            yield node
            
def split_data_coco(f_nodes, split_files):
    split_dict = defaultdict(str)
    for loc in split_files.keys():
        file = open(loc, 'r')
        file = json.load(file)     
        for f in file['data']: 
            split_dict[f['image'].split('/')[-1].split('.')[0]] = split_files[loc]    
    train = []
    val = [] 
    for idx, node in enumerate(f_nodes):
        name = node._v_name.replace('coco_', '')
        if split_dict[name] == 'train':
            train.append(node)
        elif split_dict[name] == 'val':
            val.append(node)    
    return train, val
     
            
meta_loc = '/home/danny/Downloads/'
split_files = {meta_loc + 'SpokenCOCO_train.json': 'train',
               meta_loc + 'SpokenCOCO_val.json': 'val'
               }

feature_loc = '/media/danny/seagate_2tb/Databases/coco_features.h5'
f_nodes = [node for node in read_data(feature_loc)]
train, val = split_data_coco(f_nodes, split_files)

# extract the text from the training data file and store it as plain text
with open('/home/danny/Downloads/coco_text.txt', 'w') as file:
    for i in range(0,5):
        for node in train:
            file.write(' '.join([x.decode() for x in node.tokens._f_list_nodes()[i].read()]) + '\n')
file.close()
