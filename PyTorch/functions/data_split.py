#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:57:38 2018

@author: danny
"""

# script that loads the flickr database json file in order to split
# the data into Karpathy's  train test and validation set.
import json
import os

def split_data(f_nodes, loc):
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

def split_data_coco(f_nodes):

    train_img_path = os.path.join('/data/mscoco/train2017')
    val_img_path = os.path.join('/data/mscoco/val2017')

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

def split_snli(snli_dir):
    # list the snli files
    files = os.listdir(snli_dir)
    files.sort()
    # extract the train and test examples. (indexing based on sorted directory contents
    # do not add files to the directory)
    train = []
    for line in open(os.path.join(snli_dir, files[7])):
        train.append(json.loads(line))
        
    test = []
    for line in open(os.path.join(snli_dir, files[5])):
        test.append(json.loads(line))
            
    val = []
    for line in open(os.path.join(snli_dir, files[3])):
        val.append(json.loads(line))
    
    # extract the gold label and the two sentences for each example
    train_labels = [x['gold_label'] for x in train]
    train_sentence_1 = [x['sentence1'] for x in train]
    train_sentence_2 = [x['sentence2'] for x in train]
    train = zip(train_sentence_1, train_sentence_2, train_labels)
    
    test_labels = [x['gold_label'] for x in test]
    test_sentence_1 = [x['sentence1'] for x in test]
    test_sentence_2 = [x['sentence2'] for x in test]
    test = zip(test_sentence_1, test_sentence_2, test_labels)
    
    val_labels = [x['gold_label'] for x in val]
    val_sentence_1 = [x['sentence1'] for x in val]
    val_sentence_2 = [x['sentence2'] for x in val]
    val = zip(val_sentence_1, val_sentence_2, val_labels)
    return(list(train), list(test), list(val))