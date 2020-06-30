#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:01:05 2018
This script creates and trains a next word predictor using an RNN encoder. 
Set Bi-directional to False in the RNN config! (to prevent peaking at future 
timesteps making NWP trivial)

@author: danny
"""
from __future__ import print_function
from torch.optim import lr_scheduler

import argparse
import torch
import numpy as np
import sys
import os
import pickle
import logging

sys.path.append('../functions')

from encoders import nwp_rnn_encoder, nwp_rnn_tf_att, nwp_rnn_att
from nwp_trainer import nwp_trainer
from costum_scheduler import cyclic_scheduler

parser = argparse.ArgumentParser(description = 'Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, 
                    default = '/vol/tensusers/dmerkx/next_word_prediction/',
                    help = 'location of the training sentences')
parser.add_argument('-results_loc', type = str, 
                    default = '/vol/tensusers3/dmerkx/next_word_prediction/gru_tf/',
                    help = 'location to save the trained network parameters')
parser.add_argument('-dict_loc', type = str, 
                    default = '/vol/tensusers/dmerkx/next_word_prediction/train_indices',
                    help = 'location of dictionary mapping the vocabulary to embedding indices')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 10, 
                    help = 'batch size, default: 10')
parser.add_argument('-lr', type = float, default = 0.02, 
                    help = 'learning rate, default:0.02')
parser.add_argument('-n_epochs', type = int, default = 2, 
                    help = 'number of training epochs, default: 1')
# model ids are used for the naming of model savestates and to identify how 
# many models the script should train. Each model will start with new weights and
# a new random seed.
parser.add_argument('-model_ids', type = int, default = [x for x in range(1, 9)], 
                    help = 'list of ids for the models, use single int for a single model')
parser.add_argument('-cuda', type = bool, default = True, 
                    help = 'use cuda (gpu), default: True')
parser.add_argument('-save_states', type = list, 
                    default = [1000, 3000, 10000, 30000, 100000, 300000, 
                               1000000, 3000000, 5855670], 
                    help = '#sentences after which model parameters are saved')

parser.add_argument('-gradient_clipping', type = bool, default = False, 
                    help ='use gradient clipping, default: False')
parser.add_argument('-seed', type = list, default = 745546129, 
                    help = 'optional seed for the random components')

parser.add_argument('-param', type = str, default = 'xavier')
parser.add_argument('-bias', type = str, default = 'none')

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG)
# log the settings in the argparser
for arg, value in vars(args).items():
    logging.info(f'{arg}: {value}')
    
# if the model_ids is a single int (e.g you train only a single model)
# convert to a list for compatibility further down the line
if type(args.model_ids) == int:
    args.model_ids = [args.model_ids]

# check if cuda is available and if user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
logging.info(f'Use CUDA: {cuda}')

# check if there is a random seed and create one if needed. Use it to generate
# pairs of seeds for numpy and torch for each model to be trained
if args.seed:
    # set the seed for numpy
    np.random.seed(args.seed)
    # generate 2 seeds (for numpy and torch) for each model 
    seeds = np.random.randint(0, 2**32, [max(args.model_ids), 2])
else:
    # get a seed and log it
    seed = np.random.randint(0, 2**32)
    logging.info(f'initial random seed: {seed}')
    np.random.seed(seed)
    seeds = np.random.randint(0, 2**32, [max(args.model_ids), 2])

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

# get the size of the dictionary and add 1 for the zero or padding embedding
dict_size = len(load_obj(args.dict_loc)) + 1 
# config settings for the RNN
config = {'embed':{'n_embeddings': dict_size, 'embedding_dim': 400, 
                   'sparse': False, 'padding_idx': 0
                   }, 
          'max_len': 52,
          'rnn':{'in_size': 400, 'hidden_size': 500, 'n_layers': 1, 
                 'batch_first': True, 'bidirectional': False, 'dropout': 0
                 }, 
          'lin':{'hidden_size': 400}, 
          'att': {'in_size': 500, 'heads': 10},
          'cuda': cuda
          }

# set the seeds for numpy and torch, call before training each model
def rand_seed(seeds, model_id = 1):
    np.random.seed(seeds[model_id - 1, 0])
    torch.manual_seed(seeds[model_id - 1, 1])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load(folder, file_name):
    open_file = open(os.path.join(folder, file_name))
    line = [x for x in open_file]  
    open_file.close()
    return line  
    
train = load(args.data_loc, 'train_nwp.txt')

####################### Neural network setup ##################################
def create_model(config, args, seeds, model_id = 1, cuda = False):
    # create the network and initialise the parameters
    rand_seed(seeds, model_id)
    nwp_model = nwp_rnn_tf_att(config)
    rand_seed(seeds, model_id)
    for p in nwp_model.parameters():
        if (p.dim() > 1) & (args.param == 'he'):
            torch.nn.init.kaiming_uniform_(p, nonlinearity = 'relu')
        elif (p.dim() > 1) & (args.param == 'xavier'): 
            torch.nn.init.xavier_uniform_(p)
        #elif (p.dim() <=1) & (args.bias == 'zeros'):
        #    p.data.fill_(0)
        #elif (p.dim() <=1) & (args.bias == 'normal'):
        #    torch.nn.init.normal_(p)

    # optimiser for the network
    optimizer = torch.optim.SGD(nwp_model.parameters(), lr = args.lr, 
                                momentum = .9)
    
    # set a step learning rate that decreases the lr after 1/3, 2/3 and the 
    # full dataset.
    scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                         milestones = [int(len(train)/3), 
                                                       int(len(train)/3)*2, 
                                                       len(train)], 
                                         gamma=.5, last_epoch=-1)

    # create a trainer setting the loss function, optimizer, minibatcher and 
    # lr_scheduler
    trainer = nwp_trainer(nwp_model)
    trainer.set_model_id(model_id)
    trainer.set_dict_loc(args.dict_loc)
    trainer.set_loss(torch.nn.CrossEntropyLoss(ignore_index= 0))
    trainer.set_optimizer(optimizer)
    trainer.set_lr_scheduler(scheduler, 'cyclic')
    
    #optionally use cuda and gradient clipping
    if cuda:
        trainer.set_cuda()
    if args.gradient_clipping:
        trainer.set_gradient_clipping(0.25)

    return trainer

############################# training/test loop ##############################
# outer loop trains different models (i.e. full restart of all weights)
# and the inner loop trains multiple epoch for the same model. 
for model_id in args.model_ids: 
    trainer = create_model(config, args, seeds, model_id, cuda)
    logging.info(f'Training model nr {model_id}')
    model_parameters = filter(lambda p: p.requires_grad, 
                              trainer.encoder.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f'Model parameters: {n_params}')
    
    #print(next(trainer.encoder.parameters()))

    # run the training loop for the indicated amount of epochs 
    while trainer.epoch <= args.n_epochs:
        # Train on the train set    
        trainer.train_epoch(train, args.batch_size, args.save_states, 
                            args.results_loc)
    
        if args.gradient_clipping:
            trainer.reset_grads()
        # increase epoch#
        trainer.update_epoch()
    
    # save the gradients for each epoch, can be useful to select an initial 
    # clipping value.
    if args.gradient_clipping:
        trainer.save_gradients(args.results_loc)
    ##### hard coded for my experiment, needs fixing ########
    trainer.save_params(args.results_loc, 2*args.save_states[-1])
