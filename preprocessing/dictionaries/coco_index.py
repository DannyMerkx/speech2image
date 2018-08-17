#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:29:38 2018
load the h5 file with the training data and create a dictionary with unique indices for each word (needed for torch embedding layers)
Run this once before training your neural network and point the training script to the resulting index dictionary
@author: danny
"""
import os
import pickle
from collections import defaultdict 
import tables
# name of the token feature nodes in the h5 file
feature = 'cannonical_tokens'
# save the resulting dictionary here
dict_loc = os.path.join('/data/speech2image/preprocessing/dictionaries/')
# location of the mscoco features
data_loc = os.path.join('/prep_data/coco_features.h5')
# load data
data_file = tables.open_file(data_loc)

def data_loader(h5_file):
    for x in h5_file.root:
        for y in x:
            yield y
# save dictionary
def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
# collect all the feature nodes
f_nodes = [node for node in data_loader(data_file)]
captions = []
# read in all the captions
for x in f_nodes:
    for y in eval('x.' + feature):
        captions.append([z.decode('utf-8') for z in y.read()])

index = 1
coco_dict = defaultdict(int)
# make a dictionary with a unique index for each word in the database
# start at 1 to leave index 0 as the padding embedding
for x in captions:
    for y in x:
        coco_dict[y]
        if coco_dict[y] == 0:
            coco_dict[y] = index
            index += 1
coco_dict[''] = 0
coco_dict['<s>'] = index
coco_dict['</s>'] = index + 1
# save the dictionary
save_obj(coco_dict, os.path.join(dict_loc, 'coco_indices'))

