#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:07:27 2019

@author: danny
"""
import torch
import numpy as np
import time

from F1_score import F1_score
from costum_encoders import history_encoder, char_encoder, history_transformer, char_transformer


class SNLM_trainer():
    def __init__(self, char_config, hist_config, token_dict):
        super(SNLM_trainer, self).__init__()
        self.token_dict = token_dict
        self.char_encoder = char_encoder(char_config)
        self.hist_encoder = history_encoder(hist_config)
        
        self.scheduler = False
        self.grad_clipping = False
        self.epoch = 1
        
        self.dtype = torch.FloatTensor
        
    def set_loss(self, loss):
        self.loss = loss       
    def set_optimizer(self, optim):
        self.optimizer = optim
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler 
    def set_cuda(self):
        self.dtype = torch.cuda.FloatTensor
        self.char_encoder.cuda()
        self.hist_encoder.cuda()
    def set_epoch(self, epoch):
        self.epoch = epoch
    def update_epoch(self):
        self.epoch += 1
    def set_grad_clipping(self, clip):
        self.grad_clipping = True
        self.clip = clip
    def no_grads(self):
        for param in self.char_encoder.parameters():
            param.requires_grad = False
        for param in self.hist_encoder.parameters():
            param.requires_grad = False
    def req_grads(self):
        for param in self.char_encoder.parameters():
            param.requires_grad = True
        for param in self.hist_encoder.parameters():
            param.requires_grad = True

    # convert a batch of sentences to their character-embedding indices. 
    def token_2_index(self, batch, token_dict):
        batch_size = len(batch)
        max_len = max([len(sent) for sent in batch])
        indexed_batch = np.zeros([batch_size, max_len])
        sent_lens = []
        
        for idx, sent in enumerate(batch):
            sent_lens.append(len(sent))        
            for j, token in enumerate(sent):
                indexed_batch[idx][j] = token_dict[token]
                
        return indexed_batch, sent_lens
    # mini batch the data. Converts the sentences to embedding indices
    # and the proper torch data type. 
    def mini_batcher(self, data, batch_size, token_dict, max_len = 74, 
                     shuffle = True):
        if shuffle: 
            np.random.shuffle(data)
        for start_idx in range(0, len(data) - batch_size + 1, batch_size):
            # take a minibatch from the data, cut the sentences to the allowed max
            # length and convert them to embedding indices and torch Tensor
            excerpt = data[start_idx : start_idx + batch_size]
            excerpt = [sent.split(' ')[:max_len] for sent in excerpt]
            excerpt, sent_lens = self.token_2_index(excerpt, token_dict)
            # sort by unpadded sentence length to allow use of 
            # nn.pack_padded_sequence
            #excerpt = excerpt[np.argsort(- np.array(sent_lens))]
            #sent_lens = np.array(sent_lens)[np.argsort(- np.array(sent_lens))] 
            
            yield self.dtype(excerpt), np.array(sent_lens)
            
    def train_loop(self, data, batch_size, max_len, val, gs_data):    
        # set the networks to training mode
        self.req_grads()
        self.hist_encoder.train()
        self.char_encoder.train()
          
        self.train_loss = 0
        self.start_time = time.time()
        for i, batch in enumerate(self.mini_batcher(data, batch_size, 
                                                    self.token_dict, max_len)):
            input, l = batch
            loss = self.loss.calc_marginal_likelihood(input, l, 
                                                      self.char_encoder, 
                                                      self.hist_encoder)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm_(self.char_encoder.parameters(), 
                                              self.clip)
                torch.nn.utils.clip_grad_norm_(self.hist_encoder.parameters(), 
                                              self.clip)
                
            self.optimizer.step()
            self.train_loss += loss.data
            ###################
#            for i, batch in enumerate(self.mini_batcher(val, len(val), 
#                                                    self.token_dict, 
#                                                    max_len, shuffle = False)):
#                input, l = batch
#                segs, ret = self.loss.calc_maximum_likelihood(input, l, 
#                                                              self.char_encoder, 
#                                                              self.hist_encoder)
#            precision, recall, F1 = F1_score(gs_data, ret)
#            print(F1)
            #####################
            if self.scheduler:
                self.scheduler.step()
                
            print('training loss: ' + str(self.train_loss / (i + 1)))
        print('training took: ' + str(time.time() - self.start_time) + 's')
    def test_loop(self, data, gs_data, max_len):
    
        self.no_grads()
        self.hist_encoder.eval()
        self.char_encoder.eval()    
        
        self.test_loss = 0
        for i, batch in enumerate(self.mini_batcher(data, len(data), 
                                                    self.token_dict, 
                                                    max_len, shuffle = False)):
            input, l = batch
            loss = self.loss.calc_marginal_likelihood(input, l, 
                                                      self.char_encoder,
                                                      self.hist_encoder)
            self.test_loss += loss.data  
            
        print('validation loss: ' + str(self.test_loss/ (i + 1)))    
    
        segs, ret = self.loss.calc_maximum_likelihood(input, l, 
                                                      self.char_encoder, 
                                                      self.hist_encoder)
        precision, recall, F1 = F1_score(gs_data, ret)
        print(F1)
        return segs