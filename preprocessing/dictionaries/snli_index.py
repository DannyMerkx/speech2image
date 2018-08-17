#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:29:38 2018
load the h5 file with the training data and create a dictionary with unique indices for each word (needed for torch embedding layers)
Run this once before training your neural network and point the training script to the resulting index dictionary
@author: danny
"""
import os
import pickle
from collections import defaultdict 
import sys

sys.path.append('/data/speech2image/PyTorch/functions')
from data_split import split_snli

# save the resulting dictionary here
dict_loc = os.path.join('/data/speech2image/preprocessing/dictionaries/')
snli_dir = os.path.join('/data/snli_1.0')

# load data from the dir holding snli
train, test, val = split_snli(snli_dir, tokens = True)

# save dictionary
def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

index = 1
snli_dict = defaultdict(int)
# make a dictionary with a unique index for each word in the database
# start at 1 to leave index 0 for the padding embedding
for x in train + test + val:
    for y in range(2):
        for z in x[y]:
            snli_dict[z]
            if snli_dict[z] == 0:
                snli_dict[z] = index
                index += 1

snli_dict[''] = 0
snli_dict['<s>'] = index
snli_dict['</s>'] = index + 1
# save the dictionary
save_obj(snli_dict, os.path.join(dict_loc, 'snli_indices'))

