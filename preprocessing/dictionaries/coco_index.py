#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:29:38 2018
load the h5 file with the training data and create a dictionary with unique indices for each word (needed for torch embedding layers)
Run this once before training your neural network and point the training script to the resulting index dictionary
Filters out low occurence words and words containing numerical values to be replaced by <oov>
@author: danny
"""
import os
import pickle
from collections import defaultdict 
import tables
from spell_correct import create_spell_check_dict
from text_cleanup import remove_numerical, remove_low_occurence

# name of the token feature nodes in the h5 file
feature = 'tokens'
# location of the frequency dictionary
freq_dict_loc = os.path.join('./coco_frequency')
# save the resulting dictionary here
dict_loc = os.path.join('./')
# location of the mscoco features
data_loc = os.path.join('../../PyTorch/word_vectors/coco_features.h5')
# load data
data_file = tables.open_file(data_loc)

# function to load text features
def data_loader(h5_file):
    for x in h5_file.root:
        for y in x:
            yield y
# function to load the frequency dictionary
def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
# function to save a dictionary
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

freq_dict = load_obj(freq_dict_loc)
# start indicing the dictionary at 2 because we reserve 0 for padding and 1 for the 
# out of vocab token
index = 2
coco_dict = defaultdict(int)
# make a dictionary with a unique index for each word in the database
for cap in captions:
    # replace tokens with numerical values and low occurence tokens by oov 
    #cap = remove_low_occurence(remove_numerical(cap, '<oov>'), freq_dict, 5, '<oov>')
    for word in cap:      
        if coco_dict[word] == 0:
            coco_dict[word] = index
            index += 1

# file containg a large corpus of english words, used to spot spelling mistakes
corpus_loc = os.path.join('/data/speech2image/preprocessing/dictionaries/large.txt')
# create a dictionary of spelling corrections. I.e. for words not occuring in wordnet, 
dictionary = create_spell_check_dict(coco_dict, corpus_loc)

save_obj(dictionary, os.path.join(dict_loc, 'spell_dict'))
            
coco_dict[''] = 0
coco_dict['<oov>'] = 1
coco_dict['<s>'] = index
coco_dict['</s>'] = index + 1
# save the dictionary
save_obj(coco_dict, os.path.join(dict_loc, 'coco_indices'))