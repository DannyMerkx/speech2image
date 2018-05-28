#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:24:41 2018

@author: danny
"""
# create a dictionary for the flickr dataset in order to perform word 2 index lookup for the
# word embedding layer. Run this script once before running the training process.
import json
import pickle

caps = json.load(open('/data/speech2image/preprocessing/dataset.json'))['images']

caps = [y['tokens'] for x in caps for y in x['sentences']]

word_dict = {}
count = 1
for x in caps:
    for y in x:
        try:
            word_dict[y]

        except:
            word_dict[y] = count
            count += 1

def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

save_obj(word_dict, '/data/speech2image/Pytorch/flickr_words/word_dicts')
