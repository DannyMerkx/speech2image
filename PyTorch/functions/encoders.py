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
                           quantization_layer, res_conv)
from collections import defaultdict

#################### functions for loading word embeddings ####################
# load pickled dictionary
def load_obj(loc):
    with open(f'{loc}.pkl', 'rb') as f:
        return pickle.load(f)
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
        self.RNN = nn.GRU(input_size = rnn['input_size'], 
                          hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['num_layers'], 
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

# rnn encoder for characters and tokens
class quantized_encoder(nn.Module):
    def __init__(self, config):
        super(quantized_encoder, self).__init__()
        embed = config['embed']
        rnn= config['rnn']
        att = config ['att'] 
        self.max_len = rnn['max_len']
        self.embed = nn.Embedding(num_embeddings = embed['num_chars'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx']
                                  )
        self.RNN = nn.GRU(input_size = rnn['input_size'], 
                          hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['num_layers'], 
                          batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], 
                          dropout = rnn['dropout']
                          )
        
        self.quant = quantization_layer(1000, 2048)
        
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
        x, self.VQ_loss = self.quant(x)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
        return x
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of 
        # words occuring in the training data and the location of the embeddings.
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data) 

# rnn encoder for audio (mfcc, mbn etc.) start with convolution for temporal 
# subsampling followed by n rnn layers and (multi)attention pooling.
class audio_rnn_encoder(nn.Module):
    def __init__(self, config):
        super(audio_rnn_encoder, self).__init__()
        conv = config['conv']
        rnn= config['rnn']
        att = config ['att']
        self.max_len = rnn['max_len']
        self.Conv = nn.Conv1d(in_channels = conv['in_channels'], 
                                  out_channels = conv['out_channels'], 
                                  kernel_size = conv['kernel_size'],
                                  stride = conv['stride'], 
                                  padding = conv['padding']
                                  )
        self.RNN = nn.GRU(input_size = rnn['input_size'], 
                          hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['num_layers'], 
                          batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], 
                          dropout = rnn['dropout']
                          )
        self.att = multi_attention(in_size = att['in_size'], 
                                   hidden_size = att['hidden_size'], 
                                   n_heads = att['heads']
                                   )
        
    def forward(self, input, l):
        x = self.Conv(input)
        # correct the lengths after the convolution subsampling
        cor = lambda l, ks, stride : int((l - (ks - stride)) / stride)
        l = [cor(y, self.Conv.kernel_size[0], self.Conv.stride[0]) for y in l]
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(2, 1), l, 
                                                    batch_first = True, 
                                                    enforce_sorted = False
                                                    )
        x, hx = self.RNN(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
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

# encoder which finetunes the final layers of resnet
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

class audio_conv_encoder(nn.Module):
    def __init__(self, config):
        super(audio_conv_encoder, self).__init__()
        conv_init = config['conv_init']
        conv= config['conv']
        att = config ['att']
        self.max_len = config['max_len']
        self.Conv = nn.Conv1d(in_channels = conv_init['in_channels'], 
                                  out_channels = conv_init['out_channels'], 
                                  kernel_size = conv_init['kernel_size'],
                                  stride = conv_init['stride'], 
                                  padding = conv_init['padding']
                                  )
        self.conv1 = res_conv(in_channels = conv['in_channels'][0], 
                               out_channels = conv['out_channels'][0], 
                               ks = conv['kernel_size'][0], 
                               stride = conv['stride'][0]
                               )
        self.conv2 = res_conv(in_channels = conv['in_channels'][1], 
                               out_channels = conv['out_channels'][1], 
                               ks = conv['kernel_size'][1], 
                               stride = conv['stride'][1]
                               )
        self.conv3 = res_conv(in_channels = conv['in_channels'][2], 
                               out_channels = conv['out_channels'][2], 
                               ks = conv['kernel_size'][2], 
                               stride = conv['stride'][2]
                               )
        self.conv4 = res_conv(in_channels = conv['in_channels'][3], 
                               out_channels = conv['out_channels'][3], 
                               ks = conv['kernel_size'][3], 
                               stride = conv['stride'][3]
                               )               
        self.att = multi_attention(in_size = att['in_size'], 
                                   hidden_size = att['hidden_size'], 
                                   n_heads = att['heads']
                                   )
        
    def forward(self, input, l):
        x = self.Conv(input)
        
        x = self.conv4(self.conv3(self.conv2(self.conv1(x)))).permute(0,2,1)
        
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
        return x

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