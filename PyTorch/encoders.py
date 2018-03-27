#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:54:07 2018

@author: danny
"""

from costum_layers import RHN, attention
import torch
import torch.nn as nn

# the network for embedding the vgg16 features
class img_encoder(nn.Module):
    def __init__(self):
        super(img_encoder, self).__init__()
        self.linear_transform = nn.Linear(4096,1024)
    
    def forward(self, input):
        x = self.linear_transform(input)
        return x
    
# audio encoder as described by Harwath and Glass(2016)
class Harwath_audio_encoder(nn.Module):
    def __init__(self):
        super(Harwath_audio_encoder, self).__init__()
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
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(1024)
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

# Recurrent highway network audio encoder.
class RHN_audio_encoder(nn.Module):
    def __init__(self):
        super(RHN_audio_encoder, self).__init__()
        self.Conv2d = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (40,6), 
                                 stride = (1,2), padding = 0, groups = 1)
        self.RHN = RHN(64, 1024, 2)
        self.RHN_2 = RHN(1024, 1024, 2)
        self.RHN_3 = RHN(1024, 1024, 2)
        self.RHN_4 = RHN(1024, 1024, 2)
        self.att = attention(1024, 128, 1024)
        
    def forward(self, input):
        x = self.Conv2d(input)
        x = x.squeeze().permute(2,0,1).contiguous()
        x = self.RHN(x)
        x = self.RHN_2(x)
        x = self.RHN_3(x)
        x = self.RHN_4(x)
        print(x.data)
        x = self.att(x)
        return x

rhn = RHN_audio_encoder()
input = torch.autograd.Variable(torch.rand(3, 1, 40, 1024))
hx = torch.autograd.Variable(torch.rand(1, 3, 1024))
output = rhn(input)