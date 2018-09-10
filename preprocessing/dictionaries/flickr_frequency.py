#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:29:38 2018
Take an existing h5 file and make a dictionary containing the frequency of all the words in the flickr training set. The resulting dictionary 
could be used to create new token features for flickr where only words occuring n times in the train set are kept.
@author: danny
"""
import pickle
import os
from collections import defaultdict
import json
# path to the flickr8k dataset.json file
text_path = os.path.join('/data/flickr/dataset.json')
# save the dictionary in this folder
dict_loc = os.path.join('/data/speech2image/PyTorch/flickr_words')
# save dictionary
def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
# make a dictionary of all the flickr8k captions
text_dict = {}
txt = json.load(open(text_path))['images']
for x in txt:
    text_dict[x['filename'].split('.')[0]] = x
    
# make a dictionary containing the frequency of each word in the training part of the corpus
flickr_dict = defaultdict(int)
for x in text_dict.keys():
    # only take into account the training set of the corpus
    if text_dict[x]['split'] == 'train':
        captions = text_dict[x]['sentences']
        for y in captions:
            for z in y['tokens']:
                flickr_dict[z] += 1

save_obj(flickr_dict, os.path.join(dict_loc, 'flickr_frequency'))