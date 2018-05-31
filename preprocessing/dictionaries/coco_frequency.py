#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:17:28 2018
An attemt at spelling correction for mscoco. very conserative I take all words that
do not occur in either a large dictionary or wordnet and only look at corrections of 1 edit.
The edit has to be a word that already occurs at least once in the corpus. also create a dictionary with the frequency of each word
in mscoco
@author: danny
"""

import os
import json
import pickle
from nltk.tokenize.nist import NISTTokenizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from contractions import contractions
from spell_dict import create_spell_check_dict

# load the list of common contractions
contract = contractions()

# save dictionary
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

#path to the annotations file
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
sent_len = []
for x in val_dict.keys():
    for y in val_dict[x]:
        caption = y['caption'].lower()
        caption = rep_contractions(caption, contract)
        caption = caption.replace(' ave.', ' avenue').replace(' ave ', ' avenue ').replace('&', 'and').replace('=', '')
        caption = nist.tokenize(caption)
        captions.append(''.join([x+' ' for x in caption]))
        sent_len.append(len(caption))
        for z in caption:
            try:
                coco_dict[z] += 1
            except:
                coco_dict[z] = 1
                
for x in train_dict.keys():
    for y in train_dict[x]:
        caption = y['caption'].lower()
        caption = rep_contractions(caption, contract)
        caption = caption.replace(' ave.', ' avenue').replace(' ave ', ' avenue ').replace('&', 'and').replace('=', '')
        caption = nist.tokenize(caption)
        captions.append(''.join([x+' ' for x in caption]))
        sent_len.append(len(caption))
        for z in caption:
            try:
                coco_dict[z] += 1
            except:
                coco_dict[z] = 1

dictionary = create_spell_check_dict(coco_dict)

save_obj(dictionary, '/data/speech2image/preproccesing/dictionaries/spell_dict')
save_obj(coco_dict, '/data/speech2image/preproccesing/dictionaries/coco_dict')
 
    
#
#
#most_occuring = {}
#count = []
#
#for x in coco_dict.keys():
#    count.append(coco_dict[x])
#    if coco_dict[x] > 8000:
#        most_occuring[x] = coco_dict[x]
