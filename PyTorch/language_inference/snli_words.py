#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:26:31 2018

Train a language inference system on the stanford language inference set. Can use
models pretrained on the image captioning task to improve learning

@author: danny
"""

import sys
import argparse
import numpy as np
import time
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
sys.path.append('/data/speech2image/PyTorch/functions')

from trainer import snli_trainer
from encoders import char_gru_encoder, snli
from minibatchers import iterate_snli_tokens
from data_split import split_snli
# settings for the language inference task
parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
parser.add_argument('-snli_dir', type = str, default =  '/data/snli_1.0', help = 'location of the snli data')
parser.add_argument('-results_loc', type = str, default = '/data/speech2image/PyTorch/language_inference/results/',
                    help = 'location to save the results and network parameters')
parser.add_argument('-dict_loc', type = str, default = '/data/speech2image/preprocessing/dictionaries/snli_indices')
parser.add_argument('-glove_loc', type = str, default = '/data/SentEval-master/examples/glove.840B.300d.txt', help = 'location of pretrained glove embeddings')
parser.add_argument('-glove', type = bool, default = False, help = 'use pretrained glove embeddings, default: False')

parser.add_argument('-cap_net', type = str, default = '/data/speech2image/PyTorch/coco_char/results/caption_model.25', 
                    help = 'optional location of a pretrained language model')
parser.add_argument('-lr', type = float, default = 0.0001, help = 'learning rate, default = 0.001')
parser.add_argument('-batch_size', type = int, default = 64, help = 'mini batch ize, default = 64')
parser.add_argument('-n_epochs', type = int, default = 32, help = 'number of traning epochs, default 32')
parser.add_argument('-pre_trained', type = bool, default = False, help = 'indicates whether to load a pretrained model')
parser.add_argument('-gradient_clipping', type = bool, default = False, help ='use gradient clipping, default: False')


args = parser.parse_args()

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
# get the size of the dictionary for the embedding layer (pytorch crashes if the embedding layer is not correct for the dictionary size)
# add 1 for the zero or padding embedding
dict_size = len(load_obj(args.dict_loc)) + 3    
    
# create config dictionaries with all the parameters for your encoders
char_config = {'embed':{'num_chars': dict_size, 'embedding_dim': 300, 'sparse': False, 'padding_idx': 0}, 
               'gru':{'input_size': 300, 'hidden_size': 1024, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}}
# calculate the number of input features for the classifier. multiply by two for bidirectional networks, times 3 (concatenated
# vectors + hadamard product, + cosine distance and absolute distance.)
in_feats = (char_config['gru']['hidden_size'] * 2**char_config['gru']['bidirectional']) * char_config['att']['heads']* 3 + 2
classifier_config = {'in_feats': in_feats, 'hidden': 512, 'class': 3}

# set the script to use cuda if available
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
else:
    print('using cpu')

# load data from the dir holding snli
train, test, val = split_snli(args.snli_dir, tokens = True)

############################### Neural network setup #################################################
# create the encoder and the classifier
emb_net = char_gru_encoder(char_config)
classifier = snli(classifier_config)

# create the optimizer, loss function and the learning rate scheduler
optimizer = torch.optim.Adam(list(emb_net.parameters()) + list(classifier.parameters()), 1)

# function to create a cyclic learning rate scheduler
def create_cyclic_scheduler(max_lr, min_lr, stepsize):
    lr_lambda = lambda iteration: (max_lr - min_lr)*(0.5 * (np.cos(np.pi * (1 + (3 - 1) / stepsize * iteration)) + 1))+min_lr
    cyclic_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    # lambda function which uses the cosine function to cycle the learning rate between the given min and max rates
    # the function operates between 1 and 3 (so the cos cycles from -1 to -1 ) normalise between 0 and 1 and then press between
    # min and max lr   
    return(cyclic_scheduler)
cyclic_scheduler = create_cyclic_scheduler(max_lr = args.lr, min_lr = 1e-6, stepsize = (int(len(train)/args.batch_size)*5)*4)

trainer = snli_trainer(emb_net, classifier)
trainer.set_loss(nn.CrossEntropyLoss)
trainer.set_optimizer(optimizer)
trainer.set_token_batcher()
trainer.set_dict_loc(args.dict_loc)
trainer.set_lr_scheduler(cyclic_scheduler)

# optionally use cuda, gradient clipping and pretrained glove vectors
if cuda:
    trainer.set_cuda()
# load pretrained network if provided (don't use both pretrained and glove cause
# you will reset the embeddings learned by the pretrained network. Instead pretrain the
# net using glove vectors and do not reload them)
if args.pre_trained:
    trainer.load_cap_embedder(args.cap_net)
if args.glove:
    trainer.set_glove_embeddings(args.glove_loc)
# gradient clipping with these parameters (based the avg gradient norm for the first epoch)
# can help stabilise training in the first epoch.
if args.gradient_clipping:
    trainer.set_gradient_clipping(0.0025, 0.05)

while trainer.epoch <= args.n_epochs:
    # Train on the train set
    trainer.train_epoch(train, args.batch_size)
    #evaluate on the validation set
    trainer.test_epoch(val, args.batch_size)
    # save network parameters
    trainer.save_params(args.results_loc)  
    # print some info about this epoch
    trainer.report(args.n_epochs)
    if args.gradient_clipping:
        # I found that updating the clip value at each epoch did not work well     
        # trainer.update_clip()
        trainer.reset_grads()
    
trainer.test_epoch(test, args.batch_size)
trainer.print_test_loss()
trainer.print_accuracy()
        
# save the gradients for each epoch, can be usefull to select an initial clipping value.
if args.gradient_clipping:
    trainer.save_gradients(args.results_loc)