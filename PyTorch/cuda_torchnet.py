#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:13:00 2018

@author: danny
"""
#!/usr/bin/env python
from __future__ import print_function

import sys
import time
import numpy as np
import tables
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

from minibatchers import iterate_minibatches, iterate_minibatches_resize
from costum_loss import cosine_hinge_loss, dot_hinge_loss, l2norm_hinge_loss
from evaluate import calc_recall_at_n

## implementation of an CNN for af recognition. made for use with mel filterbank features
parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

parser.add_argument('-data_loc', type = str, default = '/prep_data/flickr_features.h5',
                    help = 'location of the feature file, default: /data/processed/fbanks.h5')
parser.add_argument('-batch_size', type = int, default = 256, help = 'batch size, default: 128')

parser.add_argument('-lr', type = float, default = 0.00005, help = 'learning rate, default:0.00005')
parser.add_argument('-n_epochs', type = int, default = 50, help = 'number of training epochs, default: 5')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')

args = parser.parse_args()

# check if cuda is availlable and user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
    # if cuda cast all variables as cuda tensor
    dtype = torch.cuda.FloatTensor
else:
    print('using cpu')
    dtype = torch.FloatTensor
    
# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 
#get a list of all the nodes in the file
f_nodes = data_file.root._f_list_nodes()
# total number of nodes (i.e. files) 
n_nodes= len(f_nodes)

# shuffle before dividing into train test and validation sets
np.random.shuffle(f_nodes)

train = f_nodes[0:7000]
val = f_nodes[7000:7500]
test = f_nodes[7500:8000]

# the network for embedding the vgg16 features
class build_img_net(nn.Module):
    def __init__(self):
        super(build_img_net, self).__init__()
        self.linear_transform = nn.Linear(4096,1024)
    
    def forward(self, input):
        x = self.linear_transform(input)
        return x

class build_audio_net(nn.Module):
    def __init__(self):
        super(build_audio_net, self).__init__()
        self.Conv2d_1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (40,5), 
                                 stride = (1,1), padding = 0, groups = 1)
        self.Pool1 = nn.MaxPool1d(kernel_size = 4, stride = 2, padding = 0, dilation = 1, 
                                  return_indices = False, ceil_mode = False)
        self.Conv1d_1 = nn.Conv1d(in_channels = 64, out_channels = 512, kernel_size = 25,
                                  stride = 1, padding = 0, groups = 1)
        self.Conv1d_2 = nn.Conv1d(in_channels = 512, out_channels = 1024, kernel_size = 25,
                                  stride = 1, padding = 0, groups = 1)
        #self.Pool2 = nn.MaxPool1d(kernel_size = 217, stride = 1, padding = 0 , dilation = 1, return_indices = False, ceil_mode = False)
        self.Pool2 = nn.AdaptiveMaxPool1d(output_size = 1, return_indices=False)
        
    def forward(self, input):
        x = self.Conv2d_1(input)
        x = x.view(x.size(0), x.size(1),x.size(3))
        x = self.Pool1(x)
        x = self.Conv1d_1(x)
        x = self.Pool1(x)
        x = self.Conv1d_2(x)
        x = self.Pool2(x)
        x = torch.squeeze(x)
        return x.view(x.size(0), x.size(1))

#####################################################

# network modules
img_net = build_img_net()
audio_net = build_audio_net()
# move graph to gpu if cuda is availlable
if cuda:
    img_net.cuda()
    audio_net.cuda()

# optimiser
optimizer = torch.optim.SGD(list(img_net.parameters())+list(audio_net.parameters()), lr= 0.00005, momentum=0.9)

# this sets the learning rate for the model to a learning rate adapted for the number of
# epochs.
def lr_decay(optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // 5))
    for groups in optimizer.param_groups:
        groups['lr'] = lr

# training routine 
def train_epoch(epoch, img_net, audio_net, optimizer, f_nodes, batch_size):
    img_net.train()
    audio_net.train()
    # perform learning rate decay
    lr_decay(optimizer, epoch)
    # for keeping track of the average loss over all batches
    train_loss = 0
    num_batches =0
    for batch in iterate_minibatches_resize(f_nodes, batch_size, 1024, shuffle = True):
        img, audio = batch
        num_batches +=1
        # convert data to pytorch variables
        img, audio = Variable(dtype(img)), Variable(dtype(audio))
        # reset the gradients of the optimiser
        optimizer.zero_grad()
        # embed the images and audio using the networks
        img_embedding = img_net(img)
        audio_embedding = audio_net(audio)
        # calculate the loss
        loss = l2norm_hinge_loss(img_embedding, audio_embedding)
        # calculate the gradients and perform the backprop step
        loss.backward()
        optimizer.step()
        # add loss to average
        train_loss += loss.data
    return train_loss/num_batches

def test_epoch(img_net, audio_net, f_nodes, batch_size):
    # set to evaluation mode
    img_net.eval()
    audio_net.eval()
    # for keeping track of the average loss
    test_batches = 0
    test_loss = 0
    for batch in iterate_minibatches_resize(f_nodes, batch_size, 1024, shuffle = False):
        img, audio = batch 
        test_batches += 1
        # convert data to pytorch variables
        img, audio = Variable(dtype(img)), Variable(dtype(audio))
        # embed the images and audio using the networks
        img_embedding = img_net(img)
        audio_embedding = audio_net(audio)
        loss = l2norm_hinge_loss(img_embedding, audio_embedding)
        # add loss to average
        test_loss += loss.data 
    return test_loss/test_batches 
epoch = 1
# run the training loop for the indicated amount of epochs 
while epoch <= args.n_epochs:
    # keep track of runtime
    start_time = time.time()

    print('training epoch: ' + str(epoch))
    # Train on the train set
    train_loss = train_epoch(epoch, img_net, audio_net, optimizer, train, args.batch_size)
    # evaluate on the validation set
    val_loss = test_epoch(img_net, audio_net, val, args.batch_size)
    
    # calculate the recall@n
    # create a minibatcher over the validation set
    iterator = iterate_minibatches_resize(val, args.batch_size,1024, shuffle = False)
    # calc recal, pass it the iterator, the embedding functions and n
    # returns the measures columnise (speech2image retrieval) and rowwise(image2speech retrieval)
    recall_col, avg_rank_col, recall_row, avg_rank_row = calc_recall_at_n(iterator, audio_net, img_net, [1, 5, 10],dtype)

    # print some info about this epoch
    print("Epoch {} of {} took {:.3f}s".format(
            epoch, args.n_epochs, time.time() - start_time))
    print("training loss:\t\t{:.6f}".format(train_loss.cpu().numpy()))
    print("validation loss:\t\t{:.6f}".format(val_loss.cpu()))
    epoch += 1
    print('recall@1 = ' + str(recall_col[0]*100) + '%')
    print('recall@5 = ' + str(recall_col[1]*100) + '%')
    print('recall@10 = ' + str(recall_col[2]*100) + '%')
    print('average rank= ' + str(avg_rank_col))

test_loss = test_epoch(img_net, audio_net, test, args.batch_size)
# calculate the recall@n
# create a minibatcher over the test set
iterator = iterate_minibatches_resize(test, args.batch_size, 1024, shuffle = False)
# calc recal, pass it the iterator, the embedding functions and n
# returns the measures columnise (speech2image retrieval) and rowwise(image2speech retrieval)
recall_col, avg_rank_col, recall_row, avg_rank_row = calc_recall_at_n(iterator, audio_net, img_net, [1, 5, 10],dtype)

print("test loss:\t\t{:.6f}".format(test_loss.cpu()))
print('test recall@1 = ' + str(recall_col[0]*100) + '%')
print('test recall@5 = ' + str(recall_col[1]*100) + '%')
print('test recall@10 = ' + str(recall_col[2]*100) + '%')
print('test average rank= ' + str(avg_rank_col))

