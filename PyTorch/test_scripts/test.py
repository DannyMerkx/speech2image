#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:24:35 2018

@author: danny

Loads pretrained models and calculates the validation and test scores for both annotation and image retrieval. 
"""
#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
import torch
import sys
sys.path.append('../functions')
sys.path.append('../training_scripts')
from trainer import flickr_trainer
from encoder_configs import create_encoders
from minibatchers import FlickrDataset, PlacesDataset, CocoDataset
from costum_loss import batch_hinge_loss, ordered_loss, attention_loss
##################################### parameter settings ##############################################

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, 
                    default = '/vol/tensusers2/dmerkx/coco/coco_features.h5',
                    help = 'location of the feature file, default: /prep_data/flickr_features.h5')
parser.add_argument('-split_loc', type = str, 
                    default = '/vol/tensusers2/dmerkx/coco/', 
                    help = 'location of the json file containing the data split information')
parser.add_argument('-results_loc', type = str, 
                    default = '/vol/tensusers2/dmerkx/results/',
                    help = 'location of the json file containing the data split information')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 100, 
                    help = 'batch size, default: 100')
parser.add_argument('-cuda', type = bool, default = True, 
                    help = 'use cuda, default: True')
# args concerning the database and which features to load
parser.add_argument('-visual', type = str, default = 'resnet', 
                    help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'mfcc', 
                    help = 'name of the node containing the audio features, default: mfcc')

args = parser.parse_args()

img_net, cap_net = create_encoders('rnn')

# flickr needs a single location of the split json file, but places and coco have seperate json 
# files for the val and train splits, you can pass a dictionary of file locations to these 
# dataloaders
split_files = {args.split_loc + 'SpokenCOCO_train.json': 'train',
               args.split_loc + 'SpokenCOCO_val.json': 'val'
               }

# open the data file
dataset = COCODataset(args.data_loc, args.visual, args.cap, split_files)

# check if cuda is available and user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
    device = 'gpu'
else:
    print('using cpu')
    device = 'cpu'
#####################################################

# list all the trained model parameters
models = os.listdir(args.results_loc)
caption_models = [x for x in models if 'caption' in x]
img_models = [x for x in models if 'image' in x]

# run the image and caption retrieval
img_models.sort()
caption_models.sort()

# create a trainer with just the evaluator for the purpose of testing a pretrained model
trainer = flickr_trainer(img_net, cap_net, args.visual, args.cap)
trainer.set_audio_batcher()
trainer.set_loss(batch_hinge_loss)
trainer.no_grads()

# optionally use cuda
if cuda:
    trainer.set_cuda()
trainer.set_evaluator([1, 5, 10])

for img, cap in zip(img_models, caption_models):

    epoch = img.split('.')[1]
    # using a cyclic learning rate I'm generally only interested in every 4th 
    # epoch
    if (int(epoch) % 4) > 0:
        continue
    # load the pretrained embedders
    trainer.load_cap_embedder(args.results_loc + cap, device)
    trainer.load_img_embedder(args.results_loc + img, device)
    
    # calculate the recall@n
    trainer.set_epoch(epoch)
    trainer.report_test(dataset)
