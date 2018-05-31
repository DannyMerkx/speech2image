#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:29:38 2018
load the h5 file with the training data and create a dictionary with unique indices for each word (needed for torch embedding layers)
@author: danny
"""
import sys
import pickle
import string

# name of the token feature nodes in the h5 file
feature = 'clean_tokens'

sys.path.append('/data/speech2image/PyTorch/functions')
import tables
from data_split import split_data

# load data
data_file = tables.open_file('/prep_data/flickr_features.h5')

def iterate_flickr(h5_file):
    for x in h5_file.root:
        yield x

# save dictionary
def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# collect all the feature nodes
f_nodes = [node for node in iterate_flickr(data_file)]
captions = []

for x in f_nodes:
    for y in eval('x.' + feature):
        captions.append([z.decode('utf-8') for z in y.read()])

flickr_dict = {}
index = 1
# make a dictionary with a unique index for each word in the database
# start at 1 to leave index 0 for the padding embedding
for x in captions:
    for y in x:
        try:
            flickr_dict[y]
        except:
            flickr_dict[y] = index
            index += 1

# save the dictionary
save_obj(flickr_dict, '/data/speech2image/preprocessing/dictionaries/flickr_indices')

