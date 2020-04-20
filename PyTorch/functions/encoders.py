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

from costum_layers import multi_attention, transformer_encoder, transformer_decoder, transformer, quantization_layer
from collections import defaultdict

#################### functions for loading word embeddings ####################
# load dictionary
def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
# make a dictionary of the glove vectors for words occuring in the training data
def make_glove_dict(glove, index_dict):
    glove_dict = defaultdict(str)
    for line in glove:
        line = line.split(' ')
        # if the word occurs in our data we add the glove vector to the dictionary
        if index_dict[line[0]] != 0:
            glove_dict[line[0]] = line[1:] 
    return glove_dict
        
def load_word_embeddings(dict_loc, embedding_loc, embeddings):  
    # load the dictionary containing the indices of the words in the training data
    index_dict = load_obj(dict_loc)
    # load the file with the pretraind glove embeddings
    glove = open(embedding_loc)
    # make the dictionary of words in the training data that have a glove vector
    glove_dict = make_glove_dict(glove, index_dict)
    # print for how many words we could load pretrained vectors
    print('found ' + str(len(glove_dict)) + ' glove vectors')
    index_dict = load_obj(dict_loc)
    # replace the random embeddings with the pretrained embeddings
    for key in glove_dict.keys():
        index = index_dict[key]
        if index == 0:
            print('found a glove vector that does not occur in the data')
        emb = torch.FloatTensor([float(x) for x in glove_dict[key]])
        embeddings[index] = emb
        
##############################image_caption_retrieval##########################

# rnn encoder for characters and tokens
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
        # optionally load pretrained word embeddings. takes the dictionary of 
        # words occuring in the training data and the location of the embeddings.
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
        
        self.quant = quantization_layer(100, 2048)
        
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
        x, self.dists = self.quant(x)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
        return x

    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of 
        # words occuring in the training data and the location of the embeddings.
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data) 

# rnn encoder for audio (mfcc, mbn etc.)
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
        cor = lambda l, ks, stride : int((l - (ks - stride)) / ks)
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

# encoder which in conjunction with embedding the images also finetunes the 
# final layers of resnet
class resnet_encoder(nn.Module):
    def __init__(self, config):
        super(resnet_encoder, self).__init__()
        self.n_layers = config['n_layers']
        linear = config['linear']
        self.norm = config['norm']
        self.linear_transform = nn.Linear(in_features = linear['in_size'], 
                                          out_features = linear['out_size']
                                          )
        nn.init.xavier_uniform_(self.linear_transform.weight.data)
        # set the layers which need to remain fixed
        resnet = list(models.resnet152(pretrained = True).children())
        self.resnet_pretrained = nn.Sequential(*resnet[:-self.n_layers])
        for p in self.resnet_pretrained.parameters():
            p.requires_grad = False
        # set the layers which need to be finetuned
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

###########################transformer architectures###########################

# transformer model which takes aligned input in two languages and learns
# to translate from language to the other. 
class translator_transformer(transformer):
    def __init__(self, config):
        super(translator_transformer, self).__init__()
        embed = config['embed']
        tf = config['tf']
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
        self.TF_dec = transformer_decoder(in_size = tf['input_size'], 
                                          fc_size = tf['fc_size'], 
                                          n_layers = tf['n_layers'], 
                                          h = tf['h']
                                          )
        self.linear = nn.Linear(embed['embedding_dim'], embed['num_chars'])
    # forward, during training give the transformer both languages
    def forward(self, enc_input, dec_input):
        out, targs = self.encoder_decoder_train(enc_input, dec_input)
        return out, targs
    # translate, during test time translate from one language to the other, 
    # works without decoder input from the target language. 
    def translate(self, enc_input, dec_input = None, beam_width = 1):
        # inherited function encoder_decoder_test implements beam search to
        # create translations of the encoder input. caution! only works on one 
        # sentence at a time, set batch size to 1 during test time. 
        candidates, preds, targs = self.encoder_decoder_test(enc_input, 
                                                             dec_input, 
                                                             self.max_len, 
                                                             beam_width
                                                             )
        return candidates, preds, targs

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

##############################################################################
# network concepts and experiments and networks by others
##############################################################################

# simple encoder that just sums the word embeddings of the tokens
class bow_encoder(nn.Module):
    def __init__(self, config):
        super(bow_encoder, self).__init__()
        embed = config['embed']
        self.embed = nn.Embedding(num_embeddings = embed['num_chars'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
    def forward(self, input, l):
        # embedding layers expect Long tensors
        x = self.embed(input.long())
        return x.sum(2)
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of 
        # words occuring in the training data and the location of the 
        # embeddings.
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data) 

# the convolutional character encoder described by Wehrmann et al. 
class conv_encoder(nn.Module):
    def __init__(self):
        super(conv_encoder, self).__init__()
        self.Conv1d_1 = nn.Conv1d(in_channels = 20, out_channels = 512, 
                                  kernel_size = 7, stride = 1, padding = 3, 
                                  groups = 1
                                  )
        self.Conv1d_2 = nn.Conv1d(in_channels = 512, out_channels = 512, 
                                  kernel_size = 5, stride = 1, padding = 2, 
                                  groups = 1
                                  )
        self.Conv1d_3 = nn.Conv1d(in_channels = 512, out_channels = 512, 
                                  kernel_size = 3, stride = 1, padding = 1, 
                                  groups = 1
                                  )
        self.relu = nn.ReLU()
        self.embed = nn.Embedding(num_embeddings = 100, embedding_dim = 20,
                                  sparse = False, padding_idx = 0
                                  )
        self.Pool = nn.AdaptiveMaxPool1d(output_size = 1, 
                                         return_indices = False
                                         )
        self.linear = nn.Linear(in_features = 512, out_features = 512)
    def forward(self, input, l):
        x = self.embed(input.long()).permute(0, 2, 1)
        x = self.relu(self.Conv1d_1(x))
        x = self.relu(self.Conv1d_2(x))
        x = self.relu(self.Conv1d_3(x))
        x = self.linear(self.Pool(x).squeeze())
        return nn.functional.normalize(x, p = 2, dim = 1)
