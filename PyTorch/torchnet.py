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

from minibatchers import iterate_minibatches
from costum_loss import cosine_hinge_loss, dot_hinge_loss, l2norm_hinge_loss
from evaluate import calc_recall_at_n

## implementation of an CNN for af recognition. made for use with mel filterbank features
parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')
#parser.add_argument('-load_weights', type = bool, default = False, 
#                    help = 'load a pre-trained model (True), or initialise a new one (False), default: False')
#parser.add_argument('-weight_loc', type = str, default = './model.npz',
#                    help = 'location of pretrained weights, default: ./model.npz ')
parser.add_argument('-data_loc', type = str, default = '/train_data/flickr_features.h5',
                    help = 'location of the feature file, default: /data/processed/fbanks.h5')
parser.add_argument('-batch_size', type = int, default = 128, help = 'batch size, default: 128')

parser.add_argument('-lr', type = float, default = 0.00005, help = 'learning rate, default:0.00005')
#parser.add_argument('-test', type = bool, default = False, 
#                    help = 'set to True to skip training and only run the network on the test set, use in combination with a pretrained model of course, default: False')
#parser.add_argument('-feat_type', type = str, default = 'fbanks', 
#                    help = 'type of input feature, either mfcc, fbanks, freq_spectrum or raw, default: fbanks')

args = parser.parse_args()

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

import sys
import time
import numpy as np
import tables
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

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
        #self.Pool2 = nn.MaxPool1d(kernel_size = 217, stride = 1, padding = 0 , dilation = 1,
        #                         return_indices = False, ceil_mode = False)
        self.Pool2 = nn.AdaptiveMaxPool1d(output_size = 1, return_indices=False)
        
    def forward(self, input):
        x = self.Conv2d_1(input)
        print(x.size())
        x = x.view(x.size(0), x.size(1),x.size(3))
        x = self.Pool1(x)
        x = self.Conv1d_1(x)
        x = self.Pool1(x)
        x = self.Conv1d_2(x)
        x = self.Pool2(x)
        return x.view(x.size(0), x.size(1))

#data= torch.autograd.Variable(torch.rand(16,1,40,1024))
#net = audio_net()
#out= net(data)

#####################################################

# network modules
img_net = build_img_net()
audio_net = build_audio_net()
# optimiser
optimizer = torch.optim.SGD(list(img_net.parameters())+list(audio_net.parameters()), lr= 0.00005, momentum=0.9)
# this sets the learning rate for the model to a learning rate adapted for the number of
# epochs.
def lr_decay(optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // 5))
    for groups in optimizer.param_groups:
        groups['lr'] = lr
# training routine 
def train_epoch(epoch, img_net, speech_net, optimizer, f_nodes, batch_size):
    img_net.train()
    speech_net.train()
    # perform learning rate decay
    lr_decay(optimizer, epoch)
    
    for batch in iterate_minibatches(f_nodes, batch_size, shuffle = True):
        img, audio = batch
        # convert data to pytorch variables
        img, audio = Variable(img), Variable(audio)
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

def test_epoch(img_net, speech_net, f_nodes, batch_size):
    img_net.eval()
    speech_net.eval()
    for batch in iterate_minibatches(f_nodes, batch_size, shuffle = False):
        img, audio = batch 
        # convert data to pytorch variables
        img, audio = Variable(img), Variable(audio)
        # embed the images and audio using the networks
        img_embedding = img_net(img)
        audio_embedding = audio_net(audio)
        test_loss = l2norm_hinge_loss(img_embedding, audio_embedding)
 
