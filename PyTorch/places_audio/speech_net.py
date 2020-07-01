#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:13:00 2018

@author: danny
"""
#!/usr/bin/env python
from __future__ import print_function

import tables
import argparse
import torch
import sys
import numpy as np

from torch.optim import lr_scheduler
sys.path.append('../functions')

from trainer import flickr_trainer
from costum_loss import batch_hinge_loss, ordered_loss, attention_loss
from costum_scheduler import cyclic_scheduler

from data_split import split_data_places
from encoder_configs import create_encoders
##################################### parameter settings ######################

parser = argparse.ArgumentParser(description = 
                                 'Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, 
                    default = '/prep_data/flickr_features.h5',
                    help = 'location of the feature file, default: /prep_data/flickr_features.h5')
parser.add_argument('-split_loc', type = str, 
                    default = '/data/databases/flickr/dataset.json', 
                    help = 'location of the json file containing the data split information')
parser.add_argument('-results_loc', type = str, 
                    default = '/data/speech2image/PyTorch/flickr_audio/results/',
                    help = 'location to save the trained models')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 32, 
                    help = 'batch size, default: 32')
parser.add_argument('-lr', type = float, default = 0.0002, 
                    help = 'learning rate, default:0.0002')
parser.add_argument('-n_epochs', type = int, default = 32, 
                    help = 'number of training epochs, default: 32')
parser.add_argument('-cuda', type = bool, default = True,
                    help = 'use cuda, default: True')
parser.add_argument('-vq', type = bool, default = False, 
                    help = 'use vq loss, default: True')
# args concerning the database and which features to load
parser.add_argument('-visual', type = str, default = 'resnet', 
                    help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'mfcc', 
                    help = 'name of the node containing the audio features, default: mfcc')
parser.add_argument('-gradient_clipping', type = bool, default = False, 
                    help ='use gradient clipping, default: False')

args = parser.parse_args()

# create encoders using presets defined in encoder_configs
img_net, cap_net = create_encoders('rnn')

# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 

# check if cuda is availlable and user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
else:
    print('using cpu')


def read_data(h5_file):
    for x in h5_file.root:
        for y in x._f_list_nodes()
            yield y
f_nodes = [node for node in read_data(data_file)] 

# split the database into train test and validation sets. default settings
# uses the json file with the karpathy split
train, test = split_data_places(f_nodes, args.split_loc)

np.random.shuffle(train)
val = train[:1000]
train = train[1000:]
############################### Neural network setup ##########################

# Adam optimiser. I found SGD to work terribly and could not find appropriate 
# parameter settings for it.
optimizer = torch.optim.Adam(list(img_net.parameters()) + 
                             list(cap_net.parameters()), 1)

# plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', 
#                                                    factor = 0.9, 
#                                                    patience = 100, 
#                                                    threshold = 0.0001, 
#                                                    min_lr = 1e-8, 
#                                                    cooldown = 100)

#step_scheduler = lr_scheduler.StepLR(optimizer, 1000, gamma=0.1, last_epoch=-1)
cyclic_scheduler = cyclic_scheduler(max_lr = args.lr, min_lr = 1e-6, 
                                    stepsize = (int(len(train)/args.batch_size)*5)*4,
                                    optimiser = optimizer)

# create a trainer setting the loss function, optimizer, minibatcher, 
# lr_scheduler and the r@n evaluator
trainer = flickr_trainer(img_net, cap_net, args.visual, args.cap)
trainer.set_loss(batch_hinge_loss)
trainer.set_optimizer(optimizer)
trainer.set_places_batcher()
trainer.set_lr_scheduler(cyclic_scheduler, 'cyclic')
trainer.set_att_loss(attention_loss)
# if using a VQ layer, the trainer should use the VQ layers' loss 
if args.vq:
    trainer.set_VQ_loss()
# optionally use cuda, gradient clipping and pretrained glove vectors
if cuda:
    trainer.set_cuda()
trainer.set_evaluator([1, 5, 10])
# gradient clipping with these parameters (based the avg gradient norm for the 
# first epoch) can help stabilise training in the first epoch.
if args.gradient_clipping:
    trainer.set_gradient_clipping(0.0025, 0.05)
################################# training/test loop ##########################

# run the training loop for the indicated amount of epochs 
while trainer.epoch <= args.n_epochs:
    # Train on the train set
    trainer.train_epoch(train, args.batch_size)
    # save network parameters
    trainer.save_params(args.results_loc)  
    # print some info about this epoch and evaluation on the validation set
    trainer.report_training(args.n_epochs, val)

    if args.gradient_clipping:
        # I found that updating the clip value at each epoch did not work well     
        # trainer.update_clip()
        trainer.reset_grads()
    trainer.update_epoch()
trainer.report_test(test)

# save the gradients for each epoch, can be useful to select an initial 
# clipping value.
if args.gradient_clipping:
    trainer.save_gradients(args.results_loc)
