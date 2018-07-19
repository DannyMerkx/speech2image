#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:26:31 2018

Train a language inference system on the stanford language inference set. Can use
models pretrained on the image captioning task to improve learning

@author: danny
"""

import sys
import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
sys.path.append('/data/speech2image/PyTorch/functions')

from encoders import char_gru_encoder, snli
from prep_text import char_2_index
from minibatchers import iterate_snli
from data_split import split_snli

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
parser.add_argument('-snli_dir', type = str, default =  '/data/snli_1.0', help = 'location of the snli data')
parser.add_argument('-results_loc', type = str, default = '/data/speech2image/PyTorch/language_inference/results/',
                    help = 'location to save the results and network parameters')
parser.add_argument('-cap_net', type = str, default = '/data/speech2image/PyTorch/coco_char/results/caption_model.25', 
                    help = 'optional location of a pretrained model')
parser.add_argument('-lr', type = float, default = 0.0001, help = 'learning rate, default = 0.001')
parser.add_argument('-batch_size', type = int, default = 64, help = 'mini batch ize, default = 64')
parser.add_argument('-n_epochs', type = int, default = 32, help = 'number of traning epochs, default 32')
parser.add_argument('-pre_trained', type = bool, default = False, help = 'indicates whether to load a pretrained model')
args = parser.parse_args()

# create config dictionaries with all the parameters for your encoders
char_config = {'embed':{'num_chars': 100, 'embedding_dim': 20, 'sparse': False, 'padding_idx': 0}, 
               'gru':{'input_size': 20, 'hidden_size': 1024, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}}
# calculate the number of input features for the classifier. multiply by two for bidirectional networks, times 3 (concatenated
# vectors + hadamard product, + cosine distance and absolute distance.)
in_feats = (char_config['gru']['hidden_size'] * 2**char_config['gru']['bidirectional']) * 3 + 2
classifier_config = {'in_feats': in_feats, 'hidden': 512, 'class': 3}

# set the script to use cuda if available
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
    # if cuda cast all variables as cuda tensor
    dtype = torch.cuda.FloatTensor
else:
    print('using cpu')
    dtype = torch.FloatTensor

# load data
train, test, val = split_snli(args.snli_dir)

# function to save parameters in a results folder
def save_params(model, file_name, epoch):
    torch.save(model.state_dict(), args.results_loc + file_name + '.' +str(epoch))
    
# convenience function to embed a batch of sentences using the network
def embed(sent, lenght, network):
    # sort the sentences in descending order by length
    l_sort = np.argsort(- np.array(length))
    index = index[l_sort]
    length = np.array(length)[l_sort]   
    # embed the sentences
    index= Variable(dtype(index))
    emb = network(index, length)
    # reverse the sorting again such that the sentence pairs can be properly lined up again    
    emb = emb[torch.cuda.LongTensor(np.argsort(l_sort))]    
    return emb

# loss function expects indices corresponding to the softmax layer
def create_labels(labels):
    labs = []
    for x in labels:
        if x  == 'contradiction':
            l = 0
        elif x  == 'entailment':
            l = 1
        elif x  == 'neutral':
            l = 2
        else:
            l = 2
        labs.append(l)
    labs = torch.autograd.Variable(torch.cuda.LongTensor(labs), requires_grad = False)
    return labs

# create a feature vector for the classifier.
def feature_vector(sent1, sent2):
    # cosine distance
    cosine = torch.matmul(sent1, sent2.t()).diag()
    # absolute elementwise distance
    absolute = (sent1 - sent2).norm(1, dim = 1, keepdim = True)
    # element wise or hadamard product
    elem_wise = sent1 * sent2
    # concatenate the embeddings and the derived features into a single feature vector
    return torch.cat((sent1, sent2, elem_wise, absolute, cosine.unsqueeze(1)), 1)

# create network
emb_net = char_gru_encoder(char_config)
classifier = snli(classifier_config)

# use cuda if availlable
if cuda:
    cap_net.cuda()
    classifier.cuda()
# load pretrained network
if args.pre_trained:
    caption_state = torch.load(args.cap_net, map_location=lambda storage, loc: storage)
    cap_net.load_state_dict(caption_state)

def create_cyclic_scheduler(max_lr, min_lr, stepsize):
    lr_lambda = lambda iteration: (max_lr - min_lr)*(0.5 * (np.cos(np.pi * (1 + (3 - 1) / stepsize * iteration)) + 1))+min_lr
    cyclic_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    # lambda function which uses the cosine function to cycle the learning rate between the given min and max rates
    # the function operates between 1 and 3 (so the cos cycles from -1 to -1 ) normalise between 0 and 1 and then press between
    # min and max lr   
    return(cyclic_scheduler)
# create the optimizer, loss function and the learning rate scheduler
optimizer = torch.optim.Adam(list(cap_net.parameters()) + list(classifier.parameters()), 1)
cross_entropy_loss = nn.CrossEntropyLoss(ignore_index = -100)
cyclic_scheduler = create_cyclic_scheduler(max_lr = args.lr, min_lr = 1e-6, stepsize = (int(len(train)/args.batch_size)*5)*4)

# training epoch, takes epoch number, embedding networks, paired lists of sentences and their labels
def train_epoch(epoch, emb_net, classifier, data):
    global iteration  
    cap_net.train()
    classifier.train()
    train_loss = 0
    num_batches = 0 
    # iterator returns converted sentences and their lenghts and labels in minibatches
    for sen1, len1, sen2, len2, labels in iterate_snli(data, max_chars = 450, args.batch_size):
        cyclic_scheduler.step()
        iteration += 1
        num_batches += 1
        # embed the sentences using the network.
        sent1 = embed(sen1, len1, emb_net)
        sent2 = embed(sen2, len2, emb_net)
        # predict the class label using the classifier. 
        prediction = classifier(feature_vector(sent1, sent2))
        # convert the true labels of the sentence pairs to indices 
        labels = create_labels(z)
        # calculate the loss an take a training step
        loss = cross_entropy_loss(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        print(train_loss.cpu()[0]/num_batches)
    return(train_loss.cpu()[0]/num_batches)    
    
# test epoch and accuracy evaluation
def test_epoch(emb_net, classifier, data):
    emb_net.eval()
    classifier.eval()
    preds = []
    val_loss = 0
    num_batches = 0
    # iterator returns converted sentences and their lenghts and labels in minibatches
    for sen1, len1, sen2, len2, labels in iterate_snli(data, max_chars = 450, args.batch_size):
        num_batches += 1
        # embed the sentences using the network.
        sent1 = embed(sen1, len1, emb_net)
        sent2 = embed(sen2, len2, emb_net)
        # predict the class label using the classifier. 
        prediction = classifier(feature_vector(sent1, sent2))
        # convert the true labels of the sentence pairs to indices 
        labels = create_labels(z)
        # calculate the loss
        loss = cross_entropy_loss(prediction, labels)
        val_loss += loss.data       
        # get the index (i) of the predicted class
        v, i = torch.max(prediction, 1)
        # compare the predicted class to the true label and add to the predictions list
        preds.append(torch.Tensor.double(i.cpu().data == labels.cpu().data).numpy())
    # concatenate all the minibatches and calculate the accuracy of the predictions
    preds = np.concatenate(preds)
    correct = np.mean(preds) * 100 
    return val_loss.cpu()[0]/num_batches, correct

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
    save_params(img_net, 'image_model', epoch)
    save_params(cap_net, 'caption_model', epoch)
    
    print("validation loss:\t\t{:.6f}".format(val_loss))
    print("validation accuracy: " + str(val_accuracy) + '%')
    epoch += 1

test_loss, test_accuracy = test_epoch(emb_net, classifier, test)
print("Test loss:\t\t{:.6f}".format(test_loss))
print("validation accuracy: " + str(test_accuracy) + '%')