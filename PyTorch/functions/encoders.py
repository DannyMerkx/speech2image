#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:54:07 2018

@author: danny
"""

from costum_layers import RHN, attention
import torch
import torch.nn as nn

import time

# audio encoder as described by Harwath and Glass(2016)
class Harwath_audio_encoder(nn.Module):
    def __init__(self):
        super(Harwath_audio_encoder, self).__init__()
        self.Conv1d_1 = nn.Conv1d(in_channels = 40, out_channels = 64, kernel_size = 5, 
                                 stride = 1, padding = 0, groups = 1)
        self.Pool1 = nn.MaxPool1d(kernel_size = 4, stride = 2, padding = 0, dilation = 1, 
                                  return_indices = False, ceil_mode = False)
        self.Conv1d_2 = nn.Conv1d(in_channels = 64, out_channels = 512, kernel_size = 25,
                                  stride = 1, padding = 0, groups = 1)
        self.Conv1d_3 = nn.Conv1d(in_channels = 512, out_channels = 1024, kernel_size = 25,
                                  stride = 1, padding = 0, groups = 1)
        self.Pool2 = nn.AdaptiveMaxPool1d(output_size = 1, return_indices=False)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.relu(self.Pool1(self.Conv1d_1(input)))
        x = self.relu(self.Pool1(self.Conv1d_2(x)))
        x = self.relu(self.Pool2(self.Conv1d_3(x)))
        x = torch.squeeze(x)
        return nn.functional.normalize(x, p = 2, dim = 1)

# gru encoder for characters
class char_gru_encoder(nn.Module):
    def __init__(self, config):
        super(char_gru_encoder, self).__init__()
        embed = config['embed']
        gru= config['gru']
        att = config ['att'] 
        self.embed = nn.Embedding(num_embeddings = embed['num_chars'], 
                                  embedding_dim = embed['embedding_dim'], sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
        self.GRU = nn.GRU(input_size = gru['input_size'], hidden_size = gru['hidden_size'], 
                          num_layers = gru['num_layers'], batch_first = gru['batch_first'],
                          bidirectional = gru['bidirectional'], dropout = gru['dropout'])
        self.att = attention(in_size = att['in_size'], hidden_size = att['hidden_size'])
        
    def forward(self, input, l):
        # embedding layers expect Long tensors
        x = self.embed(input.long())
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = torch.nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True)
        x, hx = self.GRU(x)
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

# the convolutional character encoder described by Wehrmann et al. 
class conv_encoder(nn.Module):
    def __init__(self):
        super(conv_encoder, self).__init__()
        self.Conv1d_1 = nn.Conv1d(in_channels = 20, out_channels = 512, kernel_size = 7,
                                 stride = 1, padding = 3, groups = 1)
        self.Conv1d_2 = nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 5,
                                 stride = 1, padding = 2, groups = 1)
        self.Conv1d_3 = nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 3,
                                 stride = 1, padding = 1, groups = 1)
        self.relu = nn.ReLU()
        self.embed = nn.Embedding(num_embeddings = 100, embedding_dim = 20,
                                  sparse = False, padding_idx = 0)
        self.Pool = nn.AdaptiveMaxPool1d(output_size = 1, return_indices=False)
        self.linear = nn.Linear(in_features = 512, out_features = 512)
    def forward(self, input, l):
        x = self.embed(input.long()).permute(0,2,1)
        x = self.relu(self.Conv1d_1(x))
        x = self.relu(self.Conv1d_2(x))
        x = self.relu(self.Conv1d_3(x))
        x = self.linear(self.Pool(x).squeeze())
        return nn.functional.normalize(x, p = 2, dim = 1)


# gru encoder for audio
class audio_gru_encoder(nn.Module):
    def __init__(self, config):
        super(audio_gru_encoder, self).__init__()
        conv = config['conv']
        gru = config['gru']
        att = config['att']
        self.Conv1d = nn.Conv1d(in_channels = conv['in_channels'], out_channels = conv['out_channels'], 
                                kernel_size = conv['kernel_size'], stride = conv['stride'], 
                                padding = conv['padding'], groups = 1, bias = conv['bias'])
        nn.init.xavier_uniform(self.Conv1d.weight.data)
        self.GRU = nn.GRU(input_size = gru['input_size'], hidden_size = gru['hidden_size'], 
                          num_layers = gru['num_layers'], batch_first = gru['batch_first'],
                          bidirectional = gru['bidirectional'], dropout = gru['dropout'])
        self.att = attention(in_size = att['in_size'], hidden_size = att['hidden_size'])
        
    def forward(self, input, l):
        x = self.Conv1d(input)
        x = x.permute(0, 2, 1)
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        l = int((l-4)/2)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True)
        x, hx = self.GRU(x)
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

# the network for embedding the visual features
class img_encoder(nn.Module):
    def __init__(self, config):
        super(img_encoder, self).__init__()
        linear = config['linear']
        self.norm = config['norm']
        self.linear_transform = nn.Linear(in_features = linear['in_size'], out_features = linear['out_size'])
        nn.init.xavier_uniform(self.linear_transform.weight.data)
    def forward(self, input):
        x = self.linear_transform(input)
        if self.norm:
            return nn.functional.normalize(x, p=2, dim=1)
        else:
            return x

# combination of a convolutional network topped by a GRU for audio
class RCNN_audio_encoder(nn.Module):
    def __init__(self):
        super(RCNN_audio_encoder, self).__init__()
        self.Conv2d_1 = nn.Conv1d(in_channels = 40, out_channels = 64, kernel_size = 5, 
                                 stride = 1, padding = 2, groups = 1, bias = False)
        self.Pool1 = nn.MaxPool1d(kernel_size = 4, stride = 2, padding = 0, dilation = 1, 
                                  return_indices = False, ceil_mode = False)
        self.Conv1d_1 = nn.Conv1d(in_channels = 64, out_channels = 512, kernel_size = 25,
                                  stride = 1, padding = 12, groups = 1)
        self.Conv1d_2 = nn.Conv1d(in_channels = 512, out_channels = 1024, kernel_size = 25,
                                  stride = 1, padding = 12, groups = 1)
        self.Pool2 = nn.AdaptiveMaxPool1d(output_size = 1, return_indices=False)
        self.GRU = nn.GRU(1024, 1024, num_layers = 1, batch_first = True)
        self.att = attention(1024, 128)
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(1024)
    def forward(self, input):
        x = self.norm1(self.Conv2d_1(input))
        x = self.norm2(self.Conv1d_1(x))
        x = self.norm3(self.Conv1d_2(x))
        x = x.permute(0,2,1)
        x, hx = self.GRU(x)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

# Recurrent highway network audio encoder.
class RHN_audio_encoder(nn.Module):
    def __init__(self, batch_size):
        super(RHN_audio_encoder, self).__init__()
        self.Conv2d = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (40,6), 
                                 stride = (1,2), padding = 0, groups = 1)
        self.RHN = RHN(64, 1024, 2, batch_size)
        self.RHN_2 = RHN(1024, 1024, 2, batch_size)
        self.RHN_3 = RHN(1024, 1024, 2, batch_size)
        self.RHN_4 = RHN(1024, 1024, 2, batch_size)
        self.att = attention(1024, 128, 1024)
        
    def forward(self, input):
        x = self.Conv2d(input)
        x = x.squeeze().permute(2,0,1).contiguous()
        x = self.RHN(x)
        x = self.RHN_2(x)
        x = self.RHN_3(x)
        x = self.RHN_4(x)
        x = self.att(x)
        return x

# code for simple testing of encoders and timing

#config = {'conv':{'in_channels': 39, 'out_channels': 64, 'kernel_size': 6, 'stride': 2,
#               'padding': 0, 'bias': False}, 'gru':{'input_size': 64, 'hidden_size': 1024, 
#               'num_layers': 1, 'batch_first': True, 'bidirectional': True, 'dropout': 0}, 
#               'att':{'in_size': 2048, 'hidden_size': 128}}

##start_time = time.time()
#gru = audio_gru_encoder(config)
#input = torch.autograd.Variable(torch.rand(8, 39, 2048))
#output = gru(input, l)

#time = time.time() - start_time
#print(time)
