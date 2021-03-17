#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:13:00 2018

@author: danny
"""
#!/usr/bin/env python
from __future__ import print_function

import argparse
import torch
from torch.optim import lr_scheduler
import sys
sys.path.append('../functions')

from trainer import flickr_trainer
from costum_loss import batch_hinge_loss, ordered_loss, attention_loss
from costum_scheduler import cyclic_scheduler
from minibatchers import FlickrDataset
from encoder_configs import create_encoders
##################################### parameter settings ######################

parser = argparse.ArgumentParser(description = 
                                 'Create and run an articulatory feature classification DNN')

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
parser.add_argument('-batch_size', type = int, default = 32, 
                    help = 'batch size, default: 32')
parser.add_argument('-lr', type = float, default = 0.0002, 
                    help = 'learning rate, default:0.0002')
parser.add_argument('-n_epochs', type = int, default = 32, 
                    help = 'number of training epochs, default: 32')
parser.add_argument('-cuda', type = bool, default = True,
                    help = 'use cuda, default: True')
parser.add_argument('-vq', type = bool, default = True, 
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
img_net, cap_net = create_encoders('rnn_VQ')

# open the dataset
dataset = FlickrDataset(args.data_loc, args.visual, args.cap, args.split_loc) 
# check if cuda is availlable and user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
else:
    print('using cpu')

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
                                    stepsize = (int(len(dataset.train)/args.batch_size)*5)*4,
                                    optimiser = optimizer)

# create a trainer setting the loss function, optimizer, minibatcher, 
# lr_scheduler and the r@n evaluator
trainer = flickr_trainer(img_net, cap_net, args.visual, args.cap)
trainer.set_loss(batch_hinge_loss)
trainer.set_optimizer(optimizer)
trainer.set_audio_batcher()
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
    trainer.train_epoch(dataset, args.batch_size)
    # save network parameters
    trainer.save_params(args.results_loc)  
    # print some info about this epoch and evaluation on the validation set
    trainer.report_training(args.n_epochs, dataset)

    if args.gradient_clipping:
        # I found that updating the clip value at each epoch did not work well     
        # trainer.update_clip()
        trainer.reset_grads()
    trainer.update_epoch()
trainer.report_test(dataset)

# save the gradients for each epoch, can be useful to select an initial 
# clipping value.
if args.gradient_clipping:
    trainer.save_gradients(args.results_loc)
