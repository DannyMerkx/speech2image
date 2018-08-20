#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:15:18 2018
combines 2 dictionary of embedding indices into one large dictionary 
@author: danny
"""

# combines 2 dictionaries 
import argparse
parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')
parser.add_argument('-loc1', type = str, default = 'flickr_indices', help = 'location of the first dictionary')
parser.add_argument('-loc2', type = str, default = 'the_one_dictionary', help = 'location of the second dictionary')
parser.add_argument('-result', type = str, default = 'combined_dict', help = 'location of the combined dictionary')
args = parser.parse_args()

import pickle

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
# save dictionary
def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
dict_1 = load_obj(args.loc1)

dict_2 = load_obj(args.loc2)

dict_size = len(dict_1)

for x in dict_2.keys():
    if dict_1[x] == 0:
        dict_1[x] = dict_size
        dict_size += 1

flickr_dict[''] = 0
save_obj(dict_1, args.result)

