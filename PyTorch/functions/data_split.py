#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:57:38 2018
this script loads the pre-defined datasplits for certain datasets
@author: danny
"""


# script that loads the flickr database json file in order to split
# the flickr data into Karpathy's train test and validation set.
from nltk.tokenize.nist import NISTTokenizer
import json
import os

def split_data_flickr(f_nodes, loc):
    file = json.load(open(loc))
    split_dict = {}
    for x in file['images']:
        split_dict[x['filename'].replace('.jpg', '')] = x['split']
    
    train = []
    val = []
    test = []

    for x in f_nodes:
        name = x._v_name.replace('flickr_', '')
        if split_dict[name] == 'train':
            train.append(x)
        if split_dict[name] == 'val':
            val.append(x)    
        if split_dict[name] == 'test':
            test.append(x) 
    return train, val, test

def prep_data_places(f_nodes, loc):
    split = json.load(open(loc))
    split = [x['uttid'].replace('-', '_') for x in split['data']]
    
    for node in f_nodes:
        name = node._v_name.replace('places_', '')
        if name in split:
            node._f_setattr('train', True)
        else:
            node._f_setattr('train', False)

def split_data_places(f_nodes, loc):
    split = json.load(open(loc))
    split = [x['uttid'].replace('-', '_') for x in split['data']]
    
    train = []
    test = []
    
    for node in f_nodes:
        if node._f_getattr('train'):
            train.append(node)
        else:
            test.append(node)
    return train, test

# Karpathy's MSCOCO split
def split_data_coco(f_nodes, loc):
    train_img_path = os.path.join(loc, 'train2017')
    val_img_path = os.path.join(loc, 'val2017')

    train_imgs = os.listdir(train_img_path)
    val_imgs = os.listdir(val_img_path)
    
    # strip the files to their basename (remove file extension)
    train_imgs_base = [x.split('.')[0] for x in train_imgs]
    val_imgs_base = [x.split('.')[0] for x in val_imgs]
    # create a dictionary with image id pointing to the location of the image file
    train_img = {}
    for im in train_imgs_base:
        train_img[im.split('_')[-1][-6:]] = [im + '.jpg']
    val_img = {}
    for im in val_imgs_base:
        val_img[im.split('_')[-1][-6:]] = [im + '.jpg']

    train = []
    val = []
    for x in f_nodes:
        name = x._v_name
        if name.split('coco_')[1] in train_img.keys():
            train.append(x)
        if name.split('coco_')[1] in val_img.keys():
            val.append(x)
    return train, val
