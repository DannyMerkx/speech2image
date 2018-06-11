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
from collections import defaultdict 
from contractions import contractions
from spell_correct import create_spell_check_dict
#path to the annotations files
text_path = os.path.join('/data/mscoco/annotations')
# folder to save the resulting dictionaries
dict_loc = os.path.join('/data/speech2image/preprocessing/dictionaries/')
# file containg a large corpus of english words, used to spot spelling mistakes
corpus_loc = os.path.join('/data/speech2image/preprocessing/dictionaries/large.txt')

# load the list of common contractions
contract = contractions()
# load the list of english stop words
stop_words = stopwords.words('english')
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

# list the files in the annotations data folder (if the structure if the database is unchanged from how it was downloaded
# from the official website)
annotations = [os.path.join(text_path, x) for x in os.listdir(text_path)]
annotations.sort()
# load the validation annotations
val_cap = json.load(open(annotations[1]))
val_cap = val_cap['annotations']
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

val_dict = defaultdict(list)
for x in val_cap:
   key = str(x['image_id'])
   # pad short image ids with 0s so they match with the keys used later on
   while len(key) < 6:
       key = '0' + key
   val_dict[key] = val_dict[key] + [x]

coco_dict = defaultdict(int)
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
            coco_dict[z] += 1
           
for x in train_dict.keys():
    for y in train_dict[x]:
        caption = y['caption'].lower()
        caption = rep_contractions(caption, contract)
        caption = caption.replace(' ave.', ' avenue').replace(' ave ', ' avenue ').replace('&', 'and').replace('=', '')
        caption = nist.tokenize(caption)
        captions.append(''.join([x+' ' for x in caption]))
        sent_len.append(len(caption))
        for z in caption:
            coco_dict[z] += 1
# create a dictionary of spelling corrections. I.e. for words not occuring in wordnet, 
dictionary = create_spell_check_dict(coco_dict, corpus_loc)

save_obj(dictionary, os.path.join('spell_dict'))
save_obj(coco_dict, os.path.join(dict_loc, 'coco_dict'))