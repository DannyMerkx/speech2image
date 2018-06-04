#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:29:38 2018
Take an existing h5 file and make a dictionary containing the frequency of all the words in the flickr training set. The resulting dictionary 
could be used to create new token features for flickr where only words occuring n times in the train set are kept.
@author: danny
"""
import sys
import pickle
import string
import os
import json

sys.path.append('/data/speech2image/PyTorch/functions')

text_path = os.path.join('/data/speech2image/preprocessing/dataset.json')

text_dict = {}
txt = json.load(open(text_path))['images']
for x in txt:
    text_dict[x['filename'].split('.')[0]] = x

# save dictionary
def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

flickr_dict = {}

for x in text_dict.keys():
    if text_dict[x]['split'] == 'train':
        captions = text_dict[x]['sentences']
        for y in captions:
            for z in y['tokens']:
                try:
                    flickr_dict[z] += 1
                except:
                    flickr_dict[z] = 1


save_obj(flickr_dict, '/data/speech2image/preprocessing/dictionaries/flickr_frequency')
