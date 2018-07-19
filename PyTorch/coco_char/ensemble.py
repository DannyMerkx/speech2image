#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:24:35 2018

@author: danny

Loads pretrained network parameters and combines their predictions into an ensemble.
Works well with a cyclic learning rate because this lr causes the network to settle in 
several disctinc local minima, while these networks then can have similar test scores
they make different mistakes leading to a meaningfull ensemble. 

for this script to work put the best scoring models in the results folder. With a cyclic learning
rate cycle of 4 epochs this is every fourth epoch (except the first few)
"""
#!/usr/bin/env python
from __future__ import print_function

import os
import tables
import argparse
import torch
import sys
import numpy as np
sys.path.append('/data/speech2image/PyTorch/functions')

from minibatchers import iterate_raw_text_5fold, iterate_raw_text
from evaluate import embed_data, recall_at_n
from encoders import img_encoder, char_gru_encoder
from data_split import split_data_coco
##################################### parameter settings ##############################################

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, default = '/prep_data/coco_features.h5',
                    help = 'location of the feature file, default: /prep_data/flickr_features.h5')
parser.add_argument('-results_loc', type = str, default = '/data/speech2image/PyTorch/coco_char/ensemble_results/',
                    help = 'location of the json file containing the data split information')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 100, help = 'batch size, default: 100')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
# args concerning the database and which features to load
parser.add_argument('-data_base', type = str, default = 'coco', help = 'database to train on, default: coco')
parser.add_argument('-visual', type = str, default = 'resnet', help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'raw_text', help = 'name of the node containing the audio features, default: raw_text')

args = parser.parse_args()

# create config dictionaries with all the parameters for your encoders

char_config = {'embed':{'num_chars': 100, 'embedding_dim': 20, 'sparse': False, 'padding_idx': 0},
               'gru':{'input_size': 20, 'hidden_size': 1024, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}}

image_config = {'linear':{'in_size': 2048, 'out_size': 2048}, 'norm': True}


# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 

# check if cuda is availlable and user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
    # if cuda cast all variables as cuda tensor
    dtype = torch.cuda.FloatTensor
else:
    print('using cpu')
    dtype = torch.FloatTensor

# get a list of all the nodes in the file. h5 format takes at most 10000 leaves per node, so big
# datasets are split into subgroups at the root node 
def iterate_large_dataset(h5_file):
    for x in h5_file.root:
        for y in x:
            yield y
# flickr doesnt need to be split at the root node
def iterate_flickr(h5_file):
    for x in h5_file.root:
        yield x

if args.data_base == 'coco':
    f_nodes = [node for node in iterate_large_dataset(data_file)]
    # define the batcher type to use.
    batcher = iterate_raw_text_5fold    
elif args.data_base == 'flickr':
    f_nodes = [node for node in iterate_flickr(data_file)]
    # define the batcher type to use.
    batcher = iterate_raw_text_5fold
elif args.data_base == 'places':
    print('places has no written captions')
else:
    print('incorrect database option')
    exit()   
    
# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
train, val = split_data_coco(f_nodes)
test = train[-5000:]
# nr of caption in the test set
test_size = len(test) * 5
def recall(cap, img, at_n, c2i, i2c, prepend):
    # calculate the recall@n. Arguments are a set of nodes, the @n values, whether to do caption2image, image2caption or both
    # and a prepend string (e.g. to print validation or test in front of the results)
    if c2i:
        # create a minibatcher over the validation set
        recall, median_rank = recall_at_n(img, cap, at_n, transpose = False)
        # print some info about this epoch
        for x in range(len(recall)):
            print(prepend + ' caption2image recall@' + str(at_n[x]) + ' = ' + str(recall[x]*100) + '%')
        print(prepend + ' caption2image median rank= ' + str(median_rank))
    if i2c:
        recall, median_rank = recall_at_n(img, cap, at_n, transpose = True)
        for x in range(len(recall)):
            print(prepend + ' image2caption recall@' + str(at_n[x]) + ' = ' + str(recall[x]*100) + '%')
        print(prepend + ' image2caption median rank= ' + str(median_rank))  
#####################################################

# network modules
img_net = img_encoder(image_config)
cap_net = char_gru_encoder(char_config)

# move graph to gpu if cuda is availlable
if cuda:
    img_net.cuda()
    cap_net.cuda()

# list all the trained model parameters
models = os.listdir(args.results_loc)
caption_models = [x for x in models if 'caption' in x]
img_models = [x for x in models if 'image' in x]

# run the image and caption retrieval and create an ensemble
img_models.sort()
caption_models.sort()
caps = torch.autograd.Variable(dtype(np.zeros((test_size, 2048)))).data
imgs = torch.autograd.Variable(dtype(np.zeros((test_size, 2048)))).data
for img, cap in zip(img_models, caption_models) :
    # load pretrained model
    img_state = torch.load(args.results_loc + img)
    caption_state = torch.load(args.results_loc + cap)
    
    img_net.load_state_dict(img_state)
    cap_net.load_state_dict(caption_state)
    iterator = batcher(test, args.batch_size, args.visual, args.cap, max_chars= 260, shuffle = False)
    caption, image = embed_data(iterator, img_net, cap_net, dtype)
    print("Epoch " + img.split('.')[1])
    #print the per epoch results for the full test set and 1k test images. 
    recall(caption, image, [1, 5, 10], c2i = True, i2c = True, prepend = 'test')
    recall(caption[:5000], image[:5000], [1, 5, 10], c2i = True, i2c = True, prepend = '1k test')
    caps += caption
    imgs += image
# print the results of the ensemble
recall(caps, imgs, [1,5,10], c2i = True, i2c = True, prepend = 'test ensemble')
recall(caps[:5000], imgs[:5000], [1,5,10], c2i = True, i2c = True, prepend = '1k test ensemble')
