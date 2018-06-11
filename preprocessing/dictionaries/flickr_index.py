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
feature = 'clean_tokens'
# save the resulting dictionary here
dict_loc = os.path.join('/data/speech2image/preprocessing/dictionaries/')
# location of the flickr8k features
data_loc = os.path.join('/prep_data/flickr_features.h5')
# load data
data_file = tables.open_file(data_loc)

def data_loader(h5_file):
    for x in h5_file.root:
        yield x
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
flickr_dict = defaultdict(int)
# make a dictionary with a unique index for each word in the database
# start at 1 to leave index 0 for the padding embedding
for x in captions:
    for y in x:
        flickr_dict[y]
        if flickr_dict[y] == 0:
            flickr_dict[y] = index
            index += 1
# save the dictionary
save_obj(flickr_dict, os.path.join(dict_loc, 'flickr_indices'))

