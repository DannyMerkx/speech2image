#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:22:23 2020

@author: danny
"""
import torch
import torch.nn.functional as f
import torch.nn as nn

audio_config = {'conv':{'in_channels': 39, 'out_channels': 64, 
                        'kernel_size': 6, 'stride': 2,'padding': 0, 
                        'bias': False
                        }, 
                'rnn':{'input_size': [64, 1024], 
                       'hidden_size': [1024, 1024], 
                       'n_layers': [1,2], 'batch_first': True, 
                       'bidirectional': [True, True], 'dropout': 0, 
                       'max_len': 1024
                       }, 
                'rnn_pack':{'input_size': [2048], 'hidden_size': [1024]},
                'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1
                       },
                'VQ':{'n_layers': 1, 'n_embs': [64], 
                      'emb_dim': [2048]
                      },
                'app_order': ['rnn', 'VQ', 'rnn_pack', 'rnn'],
                }


cap_net = rnn_pack_encoder(audio_config)
input = torch.randint(10, [32, 39, 150]).float()/10

cap_net(input, [150] * 32)

class rnn_pack_encoder(nn.Module):
    def __init__(self, config):
        super(rnn_pack_encoder, self).__init__()
        conv = config['conv']
        rnn = config['rnn']
        rnn_pack = config['rnn_pack']
        VQ = config['VQ']
        att = config ['att']
        self.max_len = rnn['max_len']
        self.Conv = nn.Conv1d(in_channels = conv['in_channels'], 
                                  out_channels = conv['out_channels'], 
                                  kernel_size = conv['kernel_size'],
                                  stride = conv['stride'], 
                                  padding = conv['padding']
                                  )
        
        self.RNN = nn.ModuleList()
        for x in range(len(rnn['n_layers'])):
            self.RNN.append(nn.GRU(input_size = rnn['input_size'][x], 
                                   hidden_size = rnn['hidden_size'][x], 
                                   num_layers = rnn['n_layers'][x],
                                   batch_first = rnn['batch_first'],
                                   bidirectional = rnn['bidirectional'][x], 
                                   dropout = rnn['dropout']
                                   )
                            )
        self.RNN_pack = nn.ModuleList()
        for x in range(len(rnn_pack['input_size'])):
            self.RNN_pack.append(nn.GRUCell(input_size = rnn_pack['input_size'][x],
                                            hidden_size = rnn_pack['hidden_size'][x]
                                            )
                                 )
            
        self.VQ = nn.ModuleList()
        for x in range(VQ['n_layers']):
            self.VQ.append(VQ_EMA_layer(VQ['n_embs'][x], VQ['emb_dim'][x]))
            
        self.att = multi_attention(in_size = att['in_size'], 
                                   hidden_size = att['hidden_size'], 
                                   n_heads = att['heads']
                                   )
        # application order of the vq and conv layers. list with 1 bool per
        # VQ/res_conv layer.
        self.app_order = config['app_order']
        
    def forward(self, input, l):
        self.batch_size = input.size(0)
        # keep track of amount of rnn and vq layers applied and the VQ loss
        r, rp, v, self.VQ_loss = 0, 0, 0, 0
        x = self.Conv(input).permute(0,2,1).contiguous()
        # correct the lengths after the convolution subsampling
        cor = lambda l, ks, stride : int((l - (ks - stride)) / stride)
        l = [cor(y, self.Conv.kernel_size[0], self.Conv.stride[0]) for y in l]     
        # go through the application order list. If true apply VQ else RNN
        for i in self.app_order:
            if i == 'rnn':
                x = self.apply_rnn(x, l, r)
                r += 1
            elif i == 'VQ':
                x, VQ_loss = self.VQ[v](x)
                self.VQ_loss += VQ_loss
                self.segmentation = self.indices2segs(self.VQ[v].idx)
                v += 1            
            elif i == 'rnn_pack': 
                x, l = self.apply_rnn_pack(x, rp, self.segmentation)
                rp += 1
        
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
        return x
    
    def indices2segs(self, inds):
        # turn the indices returned by a VQ layer into a segmentation hypothesis
        inds = inds.view(self.batch_size, -1)
        # roll the indice matrix to compare each index to the previous index
        roll = inds.data.roll(1,1)
        # set the first index of the rolled matrix to -1 to account for the sent boundary
        roll[:,0] = -1   
        # compare inds to roll to detect segment boundaries and roll back.
        # the nonzero indices now indicate segment ending frames
        segs = (inds == roll).float().roll(-1, 1)
        return segs
    
    # combine rnn with packing and unpacking the sequence
    def apply_rnn(self, input, l, RNN_idx):
        input = nn.utils.rnn.pack_padded_sequence(input, l, batch_first = True, 
                                                  enforce_sorted = False
                                                  )
        x, hx = self.RNN[RNN_idx](input)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)        
        return x
    
    def apply_rnn_pack(self, input, RNN_idx, seg):
        # initial hidden state
        h = torch.zeros([self.batch_size, self.RNN[RNN_idx].hidden_size])
        # maximum sentence length after pack operation
        max_len = (seg == False).sum(1).max()
        # output tensor
        output = torch.zeros(input.size(0), max_len, 
                             self.RNN_pack[RNN_idx].hidden_size)
        # keep track of segment idxs for the pack operation
        idx = [0] * 32 
        for x in range(0, input.size(1)):
            h = self.RNN_pack[RNN_idx](input[:, x, :], h)

            # check for each sent in the batch of the current timestep is a 
            # segment end.
            for y in range(0, input.size(0)):
                if seg[y, x] == False:
                    # if time step is a seg end, place hidden state in the out-
                    # put tensor and increase the idx
                    output[y, idx[y], :] = h[y, :]
                    idx[y] += 1
            # set hidden state to 0 for segment endings
            h = seg[:, x].unsqueeze(1).expand_as(h.squeeze()) * h.squeeze()
        # idx can now serve as new sentence length for subsequent layers
        return output, idx
    
            
        