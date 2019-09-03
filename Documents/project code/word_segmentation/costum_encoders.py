#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:13:01 2019

@author: danny
"""
import torch
import torch.nn as nn
import numpy as np

from costum_layers import transformer_encoder, transformer_decoder, transformer

################################ RNNs #########################################

# encoder which provides the character generator with a representation of the
# previous characters in the sequence. 
class history_encoder(nn.Module):
    def __init__(self, config):
        super(history_encoder, self).__init__()
        
        embed = config['embed']
        rnn = config['rnn']       
        
        self.embed = nn.Embedding(num_embeddings = embed['n_embeddings'],
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
                                  
        self.rnn = nn.LSTM(input_size = rnn['input_size'], 
                           hidden_size = rnn['hidden_size'], 
                           num_layers = rnn['num_layers'], 
                           batch_first = rnn['batch_first'],
                           bidirectional = rnn['bidirectional'], 
                           dropout = rnn['dropout'])

        self.dropout = nn.Dropout(p = 0.5)
    # set the initial hidden and cell state to 0 (torch default is 
    # random floats)
    def make_hx(self, input):
        zeros = torch.zeros(self.rnn.num_layers, input.shape[0], 
                            self.rnn.hidden_size, dtype = input.dtype, 
                            device = input.device)
        return (zeros, zeros)
            
    def forward(self, input, l):
        # create the initial hidden/cell states and convert input to embeddings
        hx = self.make_hx(input)                 
        embeddings = self.dropout(self.embed(input.long()))
        
        x = nn.utils.rnn.pack_padded_sequence(embeddings, l, 
                                              batch_first = True, 
                                              enforce_sorted = False)
        x, hx = self.rnn(x, hx)
        output, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True) 
        
        return output

# encoder which calculates the probability of a sequence taking into account
# the history representation (use the history output as first input state and 
# initial hidden state)
class char_encoder(nn.Module):
    def __init__(self, config):
        super(char_encoder, self).__init__()
        
        embed = config['embed']
        rnn = config['rnn']

        self.embed = nn.Embedding(num_embeddings = embed['n_embeddings'],
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
                                  
        self.rnn = nn.LSTM(input_size = rnn['input_size'], 
                           hidden_size = rnn['hidden_size'], 
                           num_layers = rnn['num_layers'], 
                           batch_first = rnn['batch_first'],
                           bidirectional = rnn['bidirectional'], 
                           dropout = rnn['dropout'])
        # combine a linear and a logsoftmax layer to predict the next character
        self.predict = nn.Sequential(nn.Linear(in_features = rnn['hidden_size'], 
                                               out_features = embed['n_embeddings']
                                               ),
                                     nn.LogSoftmax(dim = -1)
                                     )
        # layer to create initial hidden state from history and converts size 
        # to the right size if needed
        self.hx = nn.Sequential(nn.Linear(in_features = config['hist_size'],
                                          out_features = rnn['hidden_size'] * \
                                          rnn['num_layers']
                                          ),
                                nn.Tanh()
                                )
        # do nothing with the history representation unless size is not correct
        self.hist_rep = nn.Identity()    
        if rnn['input_size'] != config['hist_size']:
            self.hist_rep = nn.Linear(in_features = config['hist_size'],
                                         out_features = rnn['input_size'])
        
        self.dropout = nn.Dropout(p = 0.5)

    # return the log-probabilities of the target tokens from the encoder output
    def target_prob(self, probs, targets):
        targ_probs = probs.gather(-1, targets.unsqueeze(-1))
        
        return targ_probs.squeeze(-1)
    # set the initial cell state to 0 and and pass the history through a 
    # linear + tanh layer to create the initial hidden state 
    def make_hx(self, hist, input):
        cell_state = torch.zeros(self.rnn.num_layers, input.shape[0], 
                                 self.rnn.hidden_size, dtype = input.dtype, 
                                 device = input.device)
        
        hidden_state = self.hx(hist).view(-1, self.rnn.num_layers, 
                                          self.rnn.hidden_size)
        # use the provided history as initial hidden state and set cell to 0   
        return (hidden_state.permute(1,0,-1), cell_state)
            
    def forward(self, input, l, hist):
        # prepare the hidden states and the history representation
        hx = self.make_hx(hist, input)  
        hist = self.hist_rep(hist)
        # padd the input to create the prediction targets. 
        self.targets = nn.functional.pad(input, [0, 1], value = 0).long()        
        embeddings = self.embed(input.long())

        # add the history encoding as the first token in the sequence
        embeddings = torch.cat([hist, embeddings], dim = 1)                              
        embeddings = self.dropout(embeddings)
        x = nn.utils.rnn.pack_padded_sequence(embeddings, [x + 1 for x in l], 
                                              batch_first = True, 
                                              enforce_sorted = False) 
        x, hx = self.rnn(x, hx)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True) 
        
        probs = self.predict(self.dropout(x))
        target_probs = self.target_prob(probs, self.targets)
        
        return probs, target_probs

############################ Transformers #####################################

# transformer model which takes aligned input in two languages and learns
# to translate from language to the other. 
class history_transformer(transformer):
    def __init__(self, config):
        super(history_transformer, self).__init__()
        embed = config['embed']
        tf = config['tf']
        self.is_cuda = config['cuda']
        self.max_len = tf['max_len']
        # create the embedding layer
        self.embed = nn.Embedding(num_embeddings = embed['n_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
        # create the positional embeddings
        self.pos_emb = self.pos_embedding(tf['max_len'], 
                                          embed['embedding_dim'])
        # create the (stacked) transformer
        self.encoder = transformer_encoder(in_size = tf['input_size'], 
                                           fc_size = tf['fc_size'], 
                                           n_layers = tf['n_layers'], 
                                           h = tf['h'])
        self.dropout = nn.Dropout(p = 0.5)
    # forward, during training give the transformer both languages
    def forward(self, input, l = None):
        # create two masks, one for the history encoder and one to pass onto 
        # the decoder
        mask = self.create_enc_mask(input)

        # retrieve embeddings for the sentence, scale the importance relative 
        # to the pos embeddings and combine with the pos embeddings
        embeddings = self.dropout(self.embed(input.long()) * \
                                  np.sqrt(self.embed.embedding_dim) + \
                                  self.pos_emb[:input.size(1), :])

        # apply the (stacked) encoder transformer
        encoded = self.encoder(embeddings, mask)
        return encoded#, mask

    
# transformer model which takes aligned input in two languages and learns
# to translate from language to the other. 
class char_transformer(transformer):
    def __init__(self, config):
        super(char_transformer, self).__init__()
        embed = config['embed']
        tf = config['tf']
        self.is_cuda = config['cuda']
        self.max_len = tf['max_len']

        # create the embedding layer
        self.embed = nn.Embedding(num_embeddings = embed['n_embeddings'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
        # create the positional embeddings
        self.pos_emb = self.pos_embedding(tf['max_len'], 
                                          embed['embedding_dim'])

        self.decoder = transformer_decoder(in_size = tf['input_size'], 
                                           fc_size = tf['fc_size'], 
                                           n_layers = tf['n_layers'], 
                                           h = tf['h'])
        # combine a linear and a logsoftmax layer to predict the next character
        self.predict = nn.Sequential(nn.Linear(in_features = tf['input_size'], 
                                               out_features = embed['n_embeddings']
                                               ),
                                     nn.LogSoftmax(dim = -1)
                                     )
        
        self.hist_rep = nn.Sequential(nn.Linear(in_features = tf['input_size'],
                                                out_features = tf['input_size']
                                                ),
                                      nn.Tanh()
                                      )
        
        self.dropout = nn.Dropout(p = 0.5)

    # return the log-probabilities of the target tokens from the encoder output
    def target_prob(self, probs, targets):
        # skip the final input and target token(padding). eos token (</w>)
        # probability is added later on in the loss calculation
        targ_probs = probs.gather(-1, targets.unsqueeze(-1))
        
        return targ_probs.squeeze(-1)
    
    # forward, during training give the transformer both languages
    def forward(self, input, l, hist, seg_start, seg_end, hist_mask):
        # padd the input to create the prediction targets.
        self.targets = nn.functional.pad(input, [0, 1], value = 0).long()
        # padd the input with a beginning of sentence token
        input = nn.functional.pad(input, [1,0], value = 1)        
        # create the input mask
        input_mask = self.create_dec_mask(input)

        embeddings = self.embed(input.long())
        
        bos = self.hist_rep(hist[:, -1:, :])

        embeddings = torch.cat([bos, embeddings[:, 1:, :]], dim = 1) * \
                     np.sqrt(self.embed.embedding_dim) + \
                     self.pos_emb[seg_start -1 : seg_end , :]
        # apply the (stacked) decoder transformer
        decoded = self.decoder(embeddings, input_mask, hist_mask, hist)

        probs = self.predict(self.dropout(decoded))        
        target_probs = self.target_prob(probs, self.targets)
        
        return probs, target_probs