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
import tables

sys.path.append('/data/speech2image/PyTorch/functions')

from encoders import char_gru_encoder, snli, img_encoder
from minibatchers import iterate_snli_tokens
from data_split import split_snli, split_data
from trainer import flickr_trainer, snli_trainer

from costum_loss import batch_hinge_loss, ordered_loss
from evaluate import evaluate
from grad_tracker import gradient_clipping

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')
# args concerning file location
parser.add_argument('-flickr_loc', type = str, default = '/prep_data/flickr_features.h5',
                    help = 'location of the flickr feature file, default: /prep_data/flickr_features.h5')
parser.add_argument('-snli_dir', type = str, default =  '/data/snli_1.0', help = 'location of the snli data')
parser.add_argument('-split_loc', type = str, default = '/data/speech2image/preprocessing/dataset.json', 
                    help = 'location of the json file containing the flickr data split information')
parser.add_argument('-results_loc', type = str, default = '/data/speech2image/PyTorch/language_inference/results/',
                    help = 'location to save the results and network parameters')
parser.add_argument('-dict_loc', type = str, default = '/data/speech2image/preprocessing/dictionaries/snli_indices', 
                    help = 'location of the dictionary containing indices for each word in the data')
parser.add_argument('-glove_loc', type = str, default = '/data/SentEval-master/examples/glove.840B.300d.txt', 
                    help = 'location of pretrained glove embeddings')
# args concerning training settings
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
parser.add_argument('-glove', type = bool, default = False, help = 'use pretrained glove embeddings, default: False')
parser.add_argument('-lr', type = float, default = 0.0001, help = 'learning rate, default = 0.001')
parser.add_argument('-batch_size', type = int, default = 64, help = 'mini batch ize, default = 64')
parser.add_argument('-n_epochs', type = int, default = 32, help = 'number of traning epochs, default 32')
parser.add_argument('-gradient_clipping', type = bool, default = False, help ='use gradient clipping, default: False')
# args concerning the database and which features to load
parser.add_argument('-visual', type = str, default = 'resnet', help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'tokens', help = 'name of the node containing the caption features, default: tokens')

args = parser.parse_args()

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
# get the size of the dictionary for the embedding layer (pytorch crashes if the embedding layer is not correct for the dictionary size)
# add 1 for the zero or padding embedding
dict_size = len(load_obj(args.dict_loc)) + 3    
   
# create config dictionaries with all the parameters for the character encoder, image encoder and the snli classifier.
char_config = {'embed':{'num_chars': dict_size, 'embedding_dim': 300, 'sparse': False, 'padding_idx': 0}, 
               'gru':{'input_size': 300, 'hidden_size': 1024, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}}
# automatically adapt the image encoder output size to the size of the caption encoder
out_size = char_config['gru']['hidden_size'] * 2**char_config['gru']['bidirectional'] * char_config['att']['heads']
image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 'norm': True}
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

def iterate_flickr(h5_file):
    for x in h5_file.root:
        yield x
# open the flickr data file
data_file = tables.open_file(args.data_loc, mode='r+') 
f_nodes = [node for node in iterate_flickr(data_file)]

# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
flickr_train, flickr_test, flickr_val = split_data(f_nodes, args.split_loc)
# load data from the dir holding snli
snli_train, snli_test, snli_val = split_snli(args.snli_dir, tokens = True)

# function to save parameters in a results folder
def save_params(model, file_name, epoch):
    torch.save(model.state_dict(), args.results_loc + file_name + '.' +str(epoch))
    
############################### Neural network setup #################################
# network modules
img_net = img_encoder(image_config)
cap_net = char_gru_encoder(char_config)
classifier = snli(classifier_config)
# load pretrained word embeddings
if args.glove:
    cap_net.load_embeddings(args.dict_loc, args.glove_loc)

# gradient clipping with these parameters (based the avg gradient norm for the first epoch)
# can help stabilise training in the first epoch. I found that gradient clipping with 
# a cutoff based on the previous epochs avg gradient is a bad idea though.
if args.gradient_clipping:
    img_clipper = gradient_clipping(clip_value = 0.0025)
    cap_clipper = gradient_clipping(clip_value = 0.05)  
    img_clipper.register_hook(img_net)
    cap_clipper.register_hook(cap_net)
    
# Adam optimiser. I found SGD to work terribly and could not find appropriate parameter settings for it.
optimizer = torch.optim.Adam(list(img_net.parameters())+list(cap_net.parameters()) + list(classifier.parameters()), 1)

# function to create a cyclic learning rate scheduler
def create_cyclic_scheduler(max_lr, min_lr, stepsize):
    lr_lambda = lambda iteration: (max_lr - min_lr)*(0.5 * (np.cos(np.pi * (1 + (3 - 1) / stepsize * iteration)) + 1))+min_lr
    cyclic_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    # lambda function which uses the cosine function to cycle the learning rate between the given min and max rates
    # the function operates between 1 and 3 (so the cos cycles from -1 to -1 ) normalise between 0 and 1 and then press between
    # min and max lr   
    return(cyclic_scheduler)
    
cyclic_scheduler = create_cyclic_scheduler(max_lr = args.lr, min_lr = 1e-6, stepsize = (int(len(flickr_train)/args.batch_size)*5)*4)

# cross entropy loss for the snli task
cross_entropy_loss = nn.CrossEntropyLoss(ignore_index = -100)
# loss for the image captioning
c2i_loss = batch_hinge_loss

flickr = flickr_trainer(img_net, cap_net, optimizer, c2i_loss, args.visual, args.cap)
flickr.set_token_batcher()
flickr.set_cap_len(50)
flickr.set_lr_scheduler(cyclic_scheduler)
flickr.set_dict_loc(args.dict_loc)
    
snli = snli_trainer(cap_net, classifier, optimizer, cross_entropy_loss)
snli.set_token_batcher()
snli.set_cap_len(85)
snli.set_lr_scheduler(cyclic_scheduler)
snli.set_dict_loc(args.dict_loc)

# use cuda if availlable
if cuda:
    flickr.set_cuda()
    snli.set_cuda()

epoch = 1
iteration = 0
while epoch <= args.n_epochs:
    # keep track of runtime
    start_time = time.time()  
    print('training epoch: ' + str(epoch))
    
    train_loss = train_epoch(epoch, emb_net, classifier, train)
    # report on training time
    print("Epoch {} of {} took {:.3f}s".format(epoch, args.n_epochs, time.time() - start_time))
    print("training loss:\t\t{:.6f}".format(train_loss))
    val_loss, val_accuracy = test_epoch(emb_net, classifier, val)
    
    # save network parameters
    save_params(emb_net, 'caption_model', epoch)
    
    print("validation loss:\t\t{:.6f}".format(val_loss))
    print("validation accuracy: " + str(val_accuracy) + '%')
    epoch += 1

test_loss, test_accuracy = test_epoch(emb_net, classifier, test)
print("Test loss:\t\t{:.6f}".format(test_loss))
print("validation accuracy: " + str(test_accuracy) + '%')