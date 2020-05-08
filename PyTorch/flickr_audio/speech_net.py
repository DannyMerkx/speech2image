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
from torch.optim import lr_scheduler
import sys
sys.path.append('../functions')

from trainer import flickr_trainer
from costum_loss import batch_hinge_loss, ordered_loss, attention_loss
from costum_scheduler import cyclic_scheduler
from encoders import (img_encoder, audio_rnn_encoder, audio_conv_encoder, 
                      quantized_encoder, conv_VQ_encoder)
from data_split import split_data_flickr
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
# args concerning the database and which features to load
parser.add_argument('-visual', type = str, default = 'resnet', 
                    help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'mfcc', 
                    help = 'name of the node containing the audio features, default: mfcc')
parser.add_argument('-gradient_clipping', type = bool, default = False, 
                    help ='use gradient clipping, default: False')

args = parser.parse_args()

# create config dictionaries with all the parameters for your encoders
audio_config = {'conv':{'in_channels': 39, 'out_channels': 64, 
                        'kernel_size': 6, 'stride': 2,'padding': 0, 
                        'bias': False
                        }, 
                'rnn':{'input_size': 64, 'hidden_size': 1024, 'num_layers': 4, 
                       'batch_first': True, 'bidirectional': True, 
                       'dropout': 0, 'max_len':512
                       }, 
               'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}
               }

audio_config = {'conv_init':{'in_channels': 39, 'out_channels': 128, 
                             'kernel_size': 1, 'stride': 1, 'padding': 0,
                             },
                'conv':{'in_channels': [128, 128, 256, 512], 
                        'out_channels': [128, 256, 512, 1024], 
                        'kernel_size': [9, 9, 9, 9], 'stride': [2, 2, 2, 2]
                        },
                'att':{'in_size': 1024, 'hidden_size': 128, 'heads': 1},
                'max_len': 1024
                }


# automatically adapt the image encoder output size to the size of the caption
# encoder
#out_size = audio_config['rnn']['hidden_size'] * 2 ** \
#           audio_config['rnn']['bidirectional'] * audio_config['att']['heads']
out_size = 1024           
image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 'norm': True}

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
        yield x
f_nodes = [node for node in read_data(data_file)] 

# split the database into train test and validation sets. default settings
# uses the json file with the karpathy split
train, test, val = split_data_flickr(f_nodes, args.split_loc)

############################### Neural network setup ##########################

# network modules
img_net = img_encoder(image_config)
cap_net = conv_VQ_encoder(audio_config)

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
trainer.set_audio_batcher()
trainer.set_lr_scheduler(cyclic_scheduler, 'cyclic')
trainer.set_att_loss(attention_loss)
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
    #evaluate on the validation set
    trainer.test_epoch(val, args.batch_size)
    # save network parameters
    trainer.save_params(args.results_loc)  
    # print some info about this epoch
    trainer.report(args.n_epochs)
    trainer.recall_at_n(val, args.batch_size, prepend = 'validation')    

    if args.gradient_clipping:
        # I found that updating the clip value at each epoch did not work well     
        # trainer.update_clip()
        trainer.reset_grads()
    #increase epoch#
    trainer.update_epoch()
trainer.test_epoch(test, args.batch_size)
trainer.print_test_loss()
# calculate the recall@n
trainer.recall_at_n(test, args.batch_size, prepend = 'test')

# save the gradients for each epoch, can be usefull to select an initial 
# clipping value.
if args.gradient_clipping:
    trainer.save_gradients(args.results_loc)
