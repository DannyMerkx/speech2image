#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:17:28 2018
Make a dictionary with the frequency of each word in the training data. 
@author: danny
"""

import os
import json
import pickle
from nltk.tokenize.nist import NISTTokenizer
from collections import defaultdict 
import string
#path to the annotation files
text_path = os.path.join('/data/mscoco/annotations')
# folder to save the resulting dictionary
dict_loc = os.path.join('/data/speech2image/PyTorch/coco_words/')

# function tosave dictionary
def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
# function to replace some common contractions with the full version
def rep_contractions(cap, contract):   
    for x in contract.keys():
        cap = cap.replace(x, contract[x])
    return(cap)
# import the NIST tokenizer    
nist = NISTTokenizer()

# list the files in the annotations data folder (if the structure if the database is unchanged from how it was downloaded
# from the official website)
annotations = [os.path.join(text_path, x) for x in os.listdir(text_path)]
annotations.sort()
# load the training annotations
train_cap = json.load(open(annotations[0]))
train_cap = train_cap['annotations']

train_dict = defaultdict(list)
for x in train_cap:
   key = str(x['image_id'])
   # pad short image ids with 0s so they match with the keys used later on
   while len(key) < 6:
       key = '0' + key
   train_dict[key] = train_dict[key] + [x]

coco_dict = defaultdict(int)
        
for x in train_dict.keys():
    for y in train_dict[x]:
        caption = y['caption'].lower()
        caption = nist.tokenize(caption)
        for z in caption:
            z = ''.join([x for x in z if not x in string.punctuation])
            if not z == []:
                coco_dict[z] += 1

save_obj(coco_dict, os.path.join(dict_loc, 'coco_frequency'))