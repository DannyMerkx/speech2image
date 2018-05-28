#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:24:41 2018

@author: danny
"""

import json
import pickle
import os 
import string

text_path = os.path.join('/data/mscoco/annotations')

annotations = [os.path.join(text_path, x) for x in os.listdir(text_path)]
annotations.sort()
# load the validation annotations
val_cap = json.load(open(annotations[1]))
val_cap = val_cap['annotations']
# load the training annotations
train_cap = json.load(open(annotations[0]))
train_cap = train_cap['annotations']

train_dict = {}   
for x in train_cap:
   key = str(x['image_id'])
   while len(key) < 6:
       key = '0' + key
   try:
       train_dict[key] = train_dict[key] + [x]
   except:
       train_dict[key] = [x]
val_dict = {}   
for x in val_cap:
   key = str(x['image_id'])
   while len(key) < 6:
       key = '0' + key
   try:
       val_dict[key] = val_dict[key] + [x]
   except:
       val_dict[key] = [x]

coco_dict = {}
captions = []
count = 1
for x in val_dict.keys():
    for y in val_dict[x]:
        caption = y['caption'].lower()
        for y in string.punctuation:
                caption = caption.replace(y, '')
        caption = caption.split(' ')
        for z in caption:
            try:
                coco_dict[z]
            except:
                coco_dict[z] = count
                count += 1
                
for x in train_dict.keys():
    for y in train_dict[x]:
        caption = y['caption'].lower()
        for y in string.punctuation:
                caption = caption.replace(y, '')
        caption = caption.split(' ')
        for z in caption:
            try:
                coco_dict[z]
            except:
                coco_dict[z] = count
                count += 1
       
def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

save_obj(coco_dict, '/data/speech2image/PyTorch/coco_word/word_dict')