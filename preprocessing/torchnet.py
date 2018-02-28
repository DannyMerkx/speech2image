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
class img_net(nn.Module):
    def __init__(self):
        super(img_net, self).__init__()
        self.linear_transform = nn.Linear(4096,1024)
    
    def forward(self, input):
        x = self.linear_transform(input)

        return x

class audio_net(nn.Module):
    def __init__(self):
        super(audio_net, self).__init__()
        self.Conv2d1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (40,5), 
                                 stride = (1,1), padding = 0, groups = 1)
        self.Pool1 = nn.MaxPool1d(kernel_size = 4, stride = 2, padding=0, dilation=1, 
                                  return_indices=False, ceil_mode=False)
    def forward(self, input):
        x = self.Conv2d1(input)
        print(x.size())
        x = x.view(x.size(0), x.size(1),x.size(3))
        x = self.Pool1(x)
        return x
    

def build_audio_net(input_var=None):
    network = lasagne.layers.InputLayer(shape = (None, 1, 40, 1024),
                                        input_var = input_var)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, 
                                         filter_size=(40, 5), stride=(1,1), 
                                         pad='valid', W = lasagne.init.GlorotUniform(),
                                         nonlinearity= lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.FlattenLayer(network,outdim=3)
    
    network = lasagne.layers.MaxPool1DLayer(network,pool_size = (4), stride = (2), ignore_border = True)  
    
    network = lasagne.layers.Conv1DLayer(network, num_filters=512, filter_size=25,
                                         stride=1, pad='same', W=lasagne.init.GlorotUniform(), 
                                         nonlinearity=lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.MaxPool1DLayer(network,pool_size = (4), stride = (2), ignore_border = True)
    
    network = lasagne.layers.Conv1DLayer(network, num_filters=1024, filter_size=25,
                                         stride=1, pad='same', W=lasagne.init.GlorotUniform(), 
                                         nonlinearity=lasagne.nonlinearities.rectify)
    
    #network = lasagne.layers.FeaturePoolLayer(network, pool_size=253, axis = 2, pool_function=theano.tensor.max)
    network = lasagne.layers.MaxPool1DLayer(network,pool_size = lasagne.layers.get_output_shape(network, input_shapes=None)[-1], stride = 1, ignore_border = True)  
    network = lasagne.layers.ReshapeLayer(network,([0],[1]))
    
    return network    
