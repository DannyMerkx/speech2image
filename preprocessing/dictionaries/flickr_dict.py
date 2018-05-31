#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:29:38 2018

@author: danny
"""
import sys
import pickle
import string

sys.path.append('/data/speech2image/PyTorch/functions')
import tables
from data_split import split_data


data_file = tables.open_file('/prep_data/flickr_features.h5')
split_loc = '/data/speech2image/preprocessing/dataset.json'
                
def iterate_flickr(h5_file):
    for x in h5_file.root:
        yield x

# save dictionary
def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


f_nodes = [node for node in iterate_flickr(data_file)]
# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
train, test, val = split_data(f_nodes, split_loc)

captions = []

for x in train:
    for y in x.tokens:
        captions.append([z.decode('utf-8') for z in y.read()])

flickr_dict = {}

for x in captions:
    for y in x:
        try:
            flickr_dict[y] += 1
        except:
            flickr_dict[y] = 1


save_obj(flickr_dict, '/data/speech2image/preprocessing/dictionaries/flickr_dict')