#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:54:07 2018
Script with all the different encoder models.
@author: danny
"""

import pickle
import torch

import torchvision.models as models
import torch.nn as nn

from costum_layers import (multi_attention, transformer_encoder, 
                           transformer_decoder, transformer, 
                           VQ_EMA_layer, res_conv)
from collections import defaultdict

#################### functions for loading word embeddings ####################
# load pickled dictionary
def load_obj(loc):
    with open(f'{loc}.pkl', 'rb') as file:
        return pickle.load(file)
# make a dictionary of pretrained vectors for words occuring in your data
def make_emb_dict(embs, index_dict):
    emb_dict = defaultdict(str)
    for line in embs:
        # format should be such that line[0] contains the word form after split
        line = line.split(' ')
        # if the word occurs in your data add the vector to the dictionary
        if index_dict[line[0]] != 0:
            emb_dict[line[0]] = line[1:] 
    return emb_dict
# requires locations of the pretrained embeddings a index dictionary of your 
# training data and the parameters of the embedding layer. 
def load_word_embeddings(dict_loc, embedding_loc, embeddings):  
    # load the dictionary containing the indices of the words in your data
    index_dict = load_obj(dict_loc)
    # load the file with the pretraind embeddings
    embs = open(embedding_loc)
    # make a dictionary of words in your data that have a pretrained vector
    emb_dict = make_emb_dict(embs, index_dict)
    # print for how many words we could load pretrained vectors
    print(f'found {len(emb_dict)} glove vectors')
    index_dict = load_obj(dict_loc)
    # replace the random embeddings with the pretrained embeddings
    for key in emb_dict.keys():
        index = index_dict[key]
        trained_emb = torch.FloatTensor([float(x) for x in emb_dict[key]])
        embeddings[index] = trained_emb
        
##############################image_caption_retrieval##########################

# rnn encoder for characters and tokens has an embedding layer follow by
# n rnn layers and an (multi)attention pooling layer
class text_rnn_encoder(nn.Module):
    def __init__(self, config):
        super(text_rnn_encoder, self).__init__()
        embed = config['embed']
        rnn= config['rnn']
        att = config['att'] 
        self.max_len = rnn['max_len']
        self.embed = nn.Embedding(num_embeddings = embed['num_chars'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx']
                                  )
        self.RNN = nn.LSTM(input_size = rnn['input_size'], 
                           hidden_size = rnn['hidden_size'], 
                           num_layers = rnn['n_layers'], 
                           batch_first = rnn['batch_first'],
                           bidirectional = rnn['bidirectional'], 
                           dropout = rnn['dropout']
                           )
        self.att = multi_attention(in_size = att['in_size'], 
                                   hidden_size = att['hidden_size'], 
                                   n_heads = att['heads']
                                   )
        
    def forward(self, input, l):
        # embedding layers expect Long tensors
        x = self.embed(input.long())
        x = torch.nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True,
                                                    enforce_sorted = False
                                                    )
        x, hx = self.RNN(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)       
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
        return x

    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. 
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data)      

# rnn encoder for audio (mfcc, mbn etc.) start with convolution for temporal 
# subsampling followed by n rnn layers and (multi)attention pooling.
# optional VQ layers can be specified.
# expects input of dimensions Batch/Features/Time. 
class audio_rnn_encoder(nn.Module):
    def __init__(self, config):
        super(audio_rnn_encoder, self).__init__()
        conv = config['conv']
        rnn = config['rnn']
        VQ = config['VQ']
        att = config ['att']
        self.max_len = rnn['max_len']
        #self.norm = nn.LayerNorm(conv['in_channels'])
        self.Conv = nn.Conv1d(in_channels = conv['in_channels'], 
                                  out_channels = conv['out_channels'], 
                                  kernel_size = conv['kernel_size'],
                                  stride = conv['stride'], 
                                  padding = conv['padding']
                                  )
        self.RNN = nn.ModuleList()
        for x in range(len(rnn['n_layers'])):
            self.RNN.append(nn.LSTM(input_size = rnn['input_size'][x], 
                                   hidden_size = rnn['hidden_size'][x], 
                                   num_layers = rnn['n_layers'][x],
                                   batch_first = rnn['batch_first'],
                                   bidirectional = rnn['bidirectional'], 
                                   dropout = rnn['dropout']
                                   )
                            )
        # VQ layers
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
        # keep track of amount of rnn and vq layers applied
        r, v, self.VQ_loss = 0, 0, 0
        x = self.Conv(input).permute(0,2,1).contiguous()
        # correct the lengths after the convolution subsampling
        cor = lambda l, ks, stride : int((l - (ks - stride)) / stride)
        l = [cor(y, self.Conv.kernel_size[0], self.Conv.stride[0]) for y in l]        
        # go through the application order list. If true apply VQ else RNN
        for i in self.app_order:
            if i:
                x, VQ_loss = self.VQ[v](x)
                self.VQ_loss += VQ_loss
                v += 1
            else: 
                x = self.apply_rnn(x, l, r)
                r += 1
                
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
        return x
    # function to pack sequences, apply RNN and unpack sequence again
    def apply_rnn(self, input, l, RNN_idx):
        input = nn.utils.rnn.pack_padded_sequence(input, l, batch_first = True, 
                                                  enforce_sorted = False
                                                  )
        x, hx = self.RNN[RNN_idx](input)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)        
        return x
    
# the network for embedding the visual features
class img_encoder(nn.Module):
    def __init__(self, config):
        super(img_encoder, self).__init__()
        linear = config['linear']
        self.norm = config['norm']
        self.linear_transform = nn.Linear(in_features = linear['in_size'], 
                                          out_features = linear['out_size']
                                          )
        nn.init.xavier_uniform_(self.linear_transform.weight.data)
    def forward(self, input):
        x = self.linear_transform(input.squeeze())
        if self.norm:
            return nn.functional.normalize(x, p=2, dim=1)
        else:
            return x

# image encoder which finetunes the final layers of resnet
class resnet_encoder(nn.Module):
    def __init__(self, config):
        super(resnet_encoder, self).__init__()
        # n_layers determines how many layers (starting from the top) you
        # want to finetune
        self.n_layers = config['n_layers']
        linear = config['linear']
        self.norm = config['norm']
        self.linear_transform = nn.Linear(in_features = linear['in_size'], 
                                          out_features = linear['out_size']
                                          )
        nn.init.xavier_uniform_(self.linear_transform.weight.data)
        # set the layers which need to remain fixed as resnet_pretrained
        resnet = list(models.resnet152(pretrained = True).children())
        self.resnet_pretrained = nn.Sequential(*resnet[:-self.n_layers])
        for p in self.resnet_pretrained.parameters():
            p.requires_grad = False
        # set the layers which need to be finetuned as resnet_tune
        self.resnet_tune = nn.Sequential(*resnet[-self.n_layers:-1])

    def forward(self, input):
        size = input.size()
        input = input.reshape(-1, size[2], size[3], size[4])
        x = self.resnet_tune(self.resnet_pretrained(input)).squeeze()
        x = x.reshape(size[0], size[1], -1)
        x = self.linear_transform(x.mean(1))
        if self.norm:
            return nn.functional.normalize(x, p = 2, dim = 1)
        else:
            return x

# this is the convolutional VQ encoder proposed by Harwath et al.
# forward function is a bit weird but allows for configuring how many VQ layers
# to use and where they are applied instead of having to hard-code this. 
class conv_VQ_encoder(nn.Module):
    def __init__(self, config):
        super(conv_VQ_encoder, self).__init__()
        conv_init = config['conv_init']
        conv= config['conv']
        att = config ['att']
        VQ = config['VQ']
        self.max_len = config['max_len']
        # initial conv over the full feature dimension
        self.Conv = nn.Conv1d(in_channels = conv_init['in_channels'], 
                                  out_channels = conv_init['out_channels'], 
                                  kernel_size = conv_init['kernel_size'],
                                  stride = conv_init['stride'], 
                                  padding = conv_init['padding']
                                  )
        # residual conv blocks
        self.res_convs = nn.ModuleList()
        for x in range(conv['n_layers']):
            self.res_convs.append(res_conv(in_ch = conv['in_channels'][x], 
                                           out_ch = conv['out_channels'][x], 
                                           ks = conv['kernel_size'][x], 
                                           stride = conv['stride'][x]
                                           )
                                  )
        # VQ layers
        self.VQ = nn.ModuleList()
        for x in range(VQ['n_layers']):
            self.VQ.append(VQ_EMA_layer(VQ['n_embs'][x], VQ['emb_dim'][x]))
            
        # final pooling over the full time dimension              
        self.att = multi_attention(in_size = att['in_size'], 
                                   hidden_size = att['hidden_size'], 
                                   n_heads = att['heads']
                                   )
        # application order of the vq and conv layers. list with 1 bool per
        # VQ/res_conv layer.
        self.app_order = config['app_order']
        
    def forward(self, input, l):
        # keep track of amount of conv and vq layers applied
        c, v, self.VQ_loss = 0, 0, 0
        x = self.Conv(input)
        # go through the application order list. If true apply VQ else res_conv
        for i in self.app_order:
            if i:
                x = x.permute(0,2,1).contiguous()
                x, VQ_loss = self.VQ[v](x)
                x = x.permute(0,2,1).contiguous()
                self.VQ_loss += VQ_loss
                v += 1
            else: 
                x = self.res_convs[c](x)
                c += 1
                
        x = x.permute(0,2,1)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
        return x
# rnn audio encoder following Havard et al.'s catplayinginthesnow paper. 
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
        # keep track of amount of rnn, rnnpack and vq layers applied and the VQ loss
        r, rp, v, self.VQ_loss = 0, 0, 0, 0
        x = self.Conv(input).permute(0,2,1).contiguous()
        # correct the lengths after the convolution subsampling
        cor = lambda l, ks, stride : int((l - (ks - stride)) / stride)
        l = [cor(y, self.Conv.kernel_size[0], self.Conv.stride[0]) for y in l]     
        # go through the application order list. 
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
                x, l = self.apply_rnn_pack(x, l, rp, self.segmentation)
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
    
    def apply_rnn_pack(self, input, l, RNN_idx, seg):
        # initial hidden state (batch, features)
        h = torch.zeros([self.batch_size, self.RNN[RNN_idx].hidden_size])
        # maximum sentence length after pack operation
        max_len = (seg == False).sum(1).max()
        # output tensor (batch, seq, features)
        output = torch.zeros(input.size(0), max_len, 
                             self.RNN_pack[RNN_idx].hidden_size)
        # keep track of segment idxs for the pack operation
        idx = [0] * self.batch_size 
        for x in range(0, input.size(1)):
            h = self.RNN_pack[RNN_idx](input[:, x, :], h)

            # check for each sent in the batch of the current timestep is a 
            # segment end.
            for y in range(0, input.size(0)):
                if seg[y, x] == False and x < l[y]:
                    # if time step is a seg end, place hidden state in the out-
                    # put tensor and increase the idx
                    output[y, idx[y], :] = h[y, :]
                    idx[y] += 1
            # set hidden state to 0 for segment endings
            h = seg[:, x].unsqueeze(1).expand_as(h.squeeze()) * h.squeeze()
        # idx can now serve as new sentence length for subsequent layers
        return output, idx

###########################transformer architectures###########################

# transformer for image-caption retrieval
class c2i_transformer(transformer):
    def __init__(self, config):
        super(c2i_transformer, self).__init__()
        embed = config['embed']
        tf= config['tf']
        self.is_cuda = config['cuda']
        self.max_len = tf['max_len']
        # create the embedding layer
        self.embed = nn.Embedding(num_embeddings = embed['num_chars'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx']
                                  )
        # create the positional embeddings
        self.pos_emb = self.pos_embedding(tf['max_len'], 
                                          embed['embedding_dim']
                                          )
        # create the (stacked) transformer
        self.TF_enc = transformer_encoder(in_size = tf['input_size'], 
                                          fc_size = tf['fc_size'], 
                                          n_layers = tf['n_layers'], 
                                          h = tf['h']
                                          )
    def forward(self, input):
        # encode the sentence using the transformer
        encoded = self.cap2im_train(input)
        # sum over the time axis and normalise the l2 norm of the embedding
        x = nn.functional.normalize(encoded.sum(1), p = 2, dim = 1)
        return x