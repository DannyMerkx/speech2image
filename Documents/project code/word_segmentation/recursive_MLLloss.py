#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:47:04 2019

@author: danny
"""

import numpy as np
from collections import defaultdict

# class implementing maximum and marginal likelihood for sentence segmentation
# models. 
class likelihood_loss():
    def __init__(self, token_dict, max_seg_len = 7):
        super(likelihood_loss, self).__init__()
        # dictionary to keep track of intermediate results
        self.dyn_dict = defaultdict(int)
        self.l = max_seg_len
        # dictionaries needed to convert segmented results back to characters
        self.token_dict = token_dict
        self.inv_dict = {token_dict[key]: key for key in token_dict.keys()}
        self.count = 0
        
        self.count_dict = defaultdict(int)
    # dynamic calculation of the marginal likelihood    
    def calc_marginal_likelihood(self, sent, t, char_net, history_net):
        ml = 0   

        for i in range(np.clip(t - self.l, a_min = 0, a_max = t - 1), t):            
            if i > 0:    
                seg = (0,i-1,i,t-1)
                # encode the history and the new segment given the history
                # and store the result in the dynamic dictionary                                
                if not self.dyn_dict[seg]:
                    hist = history_net.history(sent[:, :i])
                    output, probs  = char_net.generator(sent[:, i:], hist[:,-1:,:])
                    self.dyn_dict[seg] = probs.sum()

                self.count_dict[seg] +=1
                # enter the recursion
                ml += self.calc_marginal_likelihood(sent[:, :i], i, char_net, 
                                                    history_net
                                                    ) + self.dyn_dict[seg] 
                self.count += 1                                 
            # history size 0 is the stop condition. 
            else:

                # encode the sentence and store the result in the dynamic
                # dictionary
                if not self.dyn_dict[(0,t)]:
                    
                    self.dyn_dict[(0,t)] = char_net.generator(sent)[1].sum()
                self.count_dict[(0,t)] += 1
                    
                ml += self.dyn_dict[(0,t)]
                self.count += 1
        return ml


    def calc_max_likelihood(self, sent, t, char_net, history_net):
        # keep track of the most likely sequence sequence so far
        max_likelihood = -np.inf
        max_sequence = []
        
        for i in range(np.clip(t - self.l, a_min = 0, a_max = t - 1), t):            
            local_likelihood = 0
            
            if i > 0: 
                seg = str(sent[:, :i]) + str(sent[:, i:])
                # encode the history and the new segment given the history
                # and store the result in the dynamic dictionary   
                if not self.dyn_dict[seg]:
                    # encode the history and the new segment given the history
                    history_net(sent[:, :i])
                    char_net(sent[:, i:], history_net.hx)
                    self.dyn_dict[seg] = char_net.prob
                # enter the recursion    
                _likelihood, _sequence = self.calc_max_likelihood(sent[:, :i], 
                                                                  i, char_net, 
                                                                  history_net)
                
                local_likelihood += _likelihood + self.dyn_dict[seg]
                # keep track of the current best (sub)solution                    
                if local_likelihood >= max_likelihood:
                    max_likelihood = local_likelihood
                    max_sequence = [_sequence] + [sent[:, i:]]  
            # history size 0 is the stop condition.        
            else:
                # encode the sentence and store the result in the dynamic
                # dictionary
                if not self.dyn_dict[str(sent)]:
                    char_net(sent)
                    self.dyn_dict[str(sent)] = char_net.prob
                
                local_likelihood += self.dyn_dict[str(sent)]
                # keep track of the current best (sub)solution      
                if local_likelihood >= max_likelihood:
                    max_likelihood = local_likelihood
                    max_sequence = sent
                    
        return max_likelihood, max_sequence
    
    # convert the segmentation given by maximum likelihood to characters
    def seg_to_sent(self, seg):
        sent = []
        for idx in range(len(seg)):
            if type(seg[idx]) == list:
                sent += [''.join(self.seg_to_sent(seg[idx]))]
            else:
                sent += [''.join(self.inv_dict[int(i.numpy())] for i in seg[idx][0])]
        return sent
    
    def reset_dict(self):
        self.dyn_dict = defaultdict(int)
        self.count_dict = defaultdict(int)
    # combine dictionary reset and calculation of marginal/max likelihood
    def marginal_likelihood(self, sent, char_net, history_net):
        self.reset_dict()
        return - self.calc_marginal_likelihood(sent, sent.shape[1], char_net, history_net)
    
    def maximum_likelihood(self, sent, char_net, history_net):
        self.reset_dict()
        ml = self.calc_max_likelihood(sent, sent.shape[1], char_net, history_net)
        return ml[0], self.seg_to_sent(ml[1])