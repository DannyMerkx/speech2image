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
import argparse
import torch
import sys
import numpy as np

sys.path.append('../functions')
sys.path.append('../training_scripts')
from trainer import flickr_trainer
from encoder_configs import create_encoders
from minibatchers import FlickrDataset
from costum_loss import batch_hinge_loss, ordered_loss, attention_loss
##################################### parameter settings ##############################################

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, 
                    default = '/vol/tensusers2/dmerkx/flickr8k/flickr_features.h5',
                    help = 'location of the feature file, default: /prep_data/flickr_features.h5')
parser.add_argument('-split_loc', type = str, 
                    default = '/vol/tensusers2/dmerkx/flickr8k/dataset.json', 
                    help = 'location of the json file containing the data split information')
parser.add_argument('-results_loc', type = str, 
                    default = '/vol/tensusers2/dmerkx/results',
                    help = 'location of the json file containing the data split information')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 100, help = 'batch size, default: 100')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
# args concerning the database and which features to load
parser.add_argument('-visual', type = str, default = 'resnet', help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'mfcc', help = 'name of the node containing the audio features, default: mfcc')
parser.add_argument('-gradient_clipping', type = bool, default = True, help ='use gradient clipping, default: True')

args = parser.parse_args()

# create encoders using presets defined in encoder_configs
img_net, cap_net = create_encoders('rnn')

# open the dataset
dataset = FlickrDataset(args.data_loc, args.visual, args.cap, args.split_loc)

# check if cuda is availlable and user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
else:
    print('using cpu')
#####################################################

# create a trainer with just the evaluator for the purpose of testing a pretrained model
trainer = flickr_trainer(img_net, cap_net, args.visual, args.cap)
trainer.set_audio_batcher()
trainer.no_grads()
    
# optionally use cuda
if cuda:
    trainer.set_cuda()
trainer.set_evaluator([1, 5, 10])

# list all the trained model parameters
models = os.listdir(args.results_loc)
caption_models = [x for x in models if 'caption' in x]
img_models = [x for x in models if 'image' in x]

out_size = img_net.linear_transform.out_features
# run the image and caption retrieval and create an ensemble
img_models.sort()
caption_models.sort()
caps = torch.autograd.Variable(trainer.dtype(np.zeros((5000, out_size)))).data
imgs = torch.autograd.Variable(trainer.dtype(np.zeros((5000, out_size)))).data

for img, cap in zip(img_models, caption_models) :    
    epoch = img.split('.')[1]
    # load the pretrained embedders
    trainer.load_cap_embedder(args.results_loc + cap, device = 'cpu')
    trainer.load_img_embedder(args.results_loc + img, device = 'cpu') 
    trainer.no_grads()
    # calculate the recall@n
    trainer.set_epoch(epoch)
    trainer.recall_at_n(dataset, prepend = 'val', mode = 'test', emb = True)

    caption =  trainer.evaluator.caption_embeddings
    image = trainer.evaluator.image_embeddings

    caps += caption
    imgs += image
# print the results of the ensemble
trainer.evaluator.set_image_embeddings(imgs)
trainer.evaluator.set_caption_embeddings(caps)

trainer.evaluator.print_caption2image('test ensemble')
trainer.evaluator.print_image2caption('test ensemble')
