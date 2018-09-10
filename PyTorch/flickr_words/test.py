#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:24:35 2018

@author: danny

Loads pretrained models in the results folder and calculates the validation and test scores for botch annotation and image retrieval
"""
#!/usr/bin/env python
from __future__ import print_function

import os
import tables
import argparse
import torch
import pickle
import sys
sys.path.append('/data/speech2image/PyTorch/functions')

from trainer import flickr_trainer
from encoders import img_encoder, text_gru_encoder
from data_split import split_data
##################################### parameter settings ##############################################

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, default = '/prep_data/flickr_features.h5',
                    help = 'location of the feature file, default: /prep_data/flickr_features.h5')
parser.add_argument('-split_loc', type = str, default = '/data/flickr/dataset.json', 
                    help = 'location of the json file containing the data split information')
parser.add_argument('-results_loc', type = str, default = '/data/speech2image/PyTorch/flickr_words/results/',
                    help = 'location to save the results and network parameters')
parser.add_argument('-dict_loc', type = str, default = '/data/speech2image/PyTorch/flickr_words/word_dict')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 100, help = 'batch size, default: 32')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
# args concerning the database and which features to load
parser.add_argument('-data_base', type = str, default = 'flickr', help = 'database to train on, default: flickr')
parser.add_argument('-visual', type = str, default = 'resnet', help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'tokens', help = 'name of the node containing the caption features, default: tokens')

args = parser.parse_args()

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
# get the size of the dictionary for the embedding layer (pytorch crashes if the embedding layer is not correct for the dictionary size)
# add 1 for the zero or padding embedding
dict_size = len(load_obj(args.dict_loc)) + 1

# create config dictionaries with all the parameters for your encoders

token_config = {'embed':{'num_chars': dict_size, 'embedding_dim': 300, 'sparse': False, 'padding_idx': 0},
               'gru':{'input_size': 300, 'hidden_size': 1024, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 2047, 'hidden_size': 128, 'heads': 1}}
# automatically adapt the image encoder output size to the size of the caption encoder
out_size = char_config['gru']['hidden_size'] * 2**token_config['gru']['bidirectional'] * char_config['att']['heads']
image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 'norm': True}


# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 

# check if cuda is availlable and user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
else:
    print('using cpu')

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
elif args.data_base == 'flickr':
    f_nodes = [node for node in iterate_flickr(data_file)]
else:
    print('incorrect database option')
    exit() 
    
# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
train, test, val = split_data(f_nodes, args.split_loc)
#####################################################
# network modules
img_net = img_encoder(image_config)
cap_net = text_gru_encoder(token_config)

# list all the trained model parameters
models = os.listdir(args.results_loc)
caption_models = [x for x in models if 'caption' in x]
img_models = [x for x in models if 'image' in x]

# run the image and caption retrieval
img_models.sort()
caption_models.sort()

# create a trainer with just the evaluator for the purpose of testing a pretrained model
trainer = flickr_trainer(img_net, cap_net, args.visual, args.cap)
trainer.set_token_batcher()
# optionally use cuda
if cuda:
    trainer.set_cuda()
trainer.set_evaluator([1, 5, 10])
trainer.set_dict_loc(args.dict_loc)

for img, cap in zip(img_models, caption_models):

    epoch = img.split('.')[1]
    # load the pretrained embedders
    trainer.load_cap_embedder(args.results_loc + cap)
    trainer.load_img_embedder(args.results_loc + img)
    
    # calculate the recall@n
    trainer.set_epoch(epoch)
    trainer.recall_at_n(val, args.batch_size, prepend = 'val')
    trainer.recall_at_n(test, args.batch_size, prepend = 'test')
