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
import sys
sys.path.append('/data/speech2image/PyTorch/functions')

from trainer import flickr_trainer
from encoders import img_encoder, text_rnn_encoder
from data_split import split_data_coco
##################################### parameter settings ##############################################

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, default = '/prep_data/coco_features.h5',
                    help = 'location of the feature file, default: /prep_data/coco_features.h5')
parser.add_argument('-results_loc', type = str, default = '/data/speech2image/PyTorch/coco_char/results/',
                    help = 'location of the encoder parameters')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 100, help = 'batch size, default: 100')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
# args concerning the database and which features to load
parser.add_argument('-visual', type = str, default = 'resnet', help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'raw_text', help = 'name of the node containing the audio features, default: raw_text')

args = parser.parse_args()

# create config dictionaries with all the parameters for your encoders
char_config = {'embed':{'num_chars': 100, 'embedding_dim': 20, 'sparse': False, 'padding_idx': 0},
               'rnn':{'input_size': 20, 'hidden_size': 1024, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}}
# automatically adapt the image encoder output size to the size of the caption encoder
out_size = char_config['rnn']['hidden_size'] * 2**char_config['rnn']['bidirectional'] * char_config['att']['heads']
image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 'norm': True}

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

# get a list of all the nodes in the file.
def iterate_data(h5_file):
    for x in h5_file.root:
        for y in x:
            yield y
f_nodes = [node for node in iterate_data(data_file)]
    
# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
train, val = split_data_coco(f_nodes)
# set aside 5000 images as test set
test = train[-5000:]
train = train[:-5000]

#####################################################
# network modules
img_net = img_encoder(image_config)
cap_net = text_gru_encoder(char_config)

# list all the trained model parameters
models = os.listdir(args.results_loc)
caption_models = [x for x in models if 'caption' in x]
img_models = [x for x in models if 'image' in x]

# create a trainer with just the evaluator for the purpose of testing a pretrained model
trainer = flickr_trainer(img_net, cap_net, args.visual, args.cap)
trainer.set_raw_text_batcher()
# optionally use cuda
if cuda:
    trainer.set_cuda()
trainer.set_evaluator([1, 5, 10])

for img, cap in zip(img_models, caption_models) :
    epoch = img.split('.')[1]
    # load the pretrained embedders
    trainer.load_cap_embedder(args.results_loc + cap)
    trainer.load_img_embedder(args.results_loc + img)
    
    # calculate the recall@n
    trainer.set_epoch(epoch)
    trainer.recall_at_n(val, args.batch_size, prepend = 'validation')
    trainer.fivefold_recall_at_n('validation')
    trainer.recall_at_n(test, args.batch_size, prepend = 'test')    
    trainer.fivefold_recall_at_n('test')
