#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:47:04 2019

@author: danny
"""
import torch
import numpy as np

# class implementing maximum and marginal likelihood for sentence segmentation
# models. 
class likelihood_loss():
    def __init__(self, token_dict, max_seg_len = 7):
        super(likelihood_loss, self).__init__()
        
        self.max_seg_len = max_seg_len
        self.token_dict = token_dict
        self.logzero = -1e6
        self.eos_id = token_dict['</w>']
        self.inv_dict = {self.token_dict[key]: key for key in \
                         self.token_dict.keys()}
    def calc_likelihood(self, input, l, char_net, history_net):
        batch_size = input.size(0)
        max_len = input.size(1)
        
        # create a schedule of all segments of the maximum segment length
        schedule = []
        for seg_start in range(1, max_len -1):
            seg_len = min(self.max_seg_len, (max_len - 1) - seg_start)
            seg_end = seg_start + seg_len
            schedule.append((seg_start, seg_len, seg_end))    
        # tensor to hold all segment probabilities. The log-probability of
        # the bos token is always 0.
        probs = torch.zeros(max_len -1, self.max_seg_len, batch_size) + self.logzero
        probs[0, 0, :] = 0
        # create the history representation. 
        hist_rep = history_net(input, l)

        for seg_start, seg_len, seg_end in schedule:
            # retrieve the appropriate section of the sentence and the history 
            # representation as inputs for the generator network.
            gen_input = input[:, seg_start:seg_end]
            gen_init = hist_rep[:, seg_start -1 , :]
            #y0 = lin(decoder_init).unsqueeze(0)
        
            #decoder_h_init = torch.tanh(lin(decoder_init))
            #decoder_h_init = decoder_h_init.view(batch_size, 1, -1)
            #decoder_h_init = torch.transpose(decoder_h_init, 0, 1).contiguous()
            
            #decoder_c_init = torch.zeros_like(decoder_h_init)
            
            #decoder_init_states = (decoder_h_init, decoder_c_init)
            
            #decoder_input = self.decoder_input_dropout(decoder_input)
            
            # use sentence lengths to determine the lenghts for the current 
            # segments. 
            _l = [min(max(1, x - seg_start), seg_len) for x in l]
            dec_out, targ_probs = char_net(gen_input, _l, gen_init, seg_start, seg_end, mask)
            
            # temporary probability value used to get the probability of all 
            # possible sub-segments of the current segment. 
            _probs = torch.zeros_like(targ_probs[:,0])
            for j in range(seg_start, seg_end):
                _probs = _probs + targ_probs[:, j - seg_start]
                #if j > j_start:
                #    tmp_logpy = tmp_logpy + is_single[j, :]
                #if j == j_start + 1:
                #    tmp_logpy = tmp_logpy + is_single[j_start, :]
                probs[seg_start][j - seg_start] = _probs + \
                                                      dec_out[:, 
                                                              j - seg_start + 1, 
                                                              self.eos_id]
        return probs, max_len, batch_size
        
    def calc_marginal_likelihood(self, input, l, char_net, history_net):
        probs, max_len, batch_size = self.calc_likelihood(input, l, char_net,
                                                          history_net)

        # get the total log_probability of all possible segmentations of the 
        # input sentence. set the BOS log probability to 0
        alpha = [self.logzero for _ in range(max_len - 1)]
        
        alpha[0] = torch.zeros_like(probs[0, 0, :])

        for j_end in range(1, max_len - 1):
                logprobs = []
                for j_start in range(max(0, j_end - self.max_seg_len), j_end):                    
                    logprobs.append(alpha[j_start] + probs[j_start + 1] \
                                    [j_end - j_start - 1])
                    
                alpha[j_end] =  torch.logsumexp(torch.stack(logprobs), dim = 0)
        total_loss = 0.0
        total_length = 0
        
        alphas = torch.stack(alpha)
        index = (torch.LongTensor(l) - 2).view(1, -1)
        
        if alphas.is_cuda:
            index = index.cuda()
            
        total_loss = - torch.gather(input = alphas, dim = 0, index = index)

        #assert NLL_loss.view(-1).size(0) == batch_size
        total_length += sum(l) - 2 * batch_size
        norm_loss = total_loss.sum() / float(total_length)
        return norm_loss
    
    def calc_maximum_likelihood(self, input, l, char_net, history_net):
        probs, max_len, batch_size = self.calc_likelihood(input, l, char_net,
                                                          history_net)
        ret = []
        for i in range(batch_size):
            alpha = [self.logzero] * (l[i] - 1)
            prev = [-1]*(l[i] - 1)
            alpha[0] = 0.0
            for j_end in range(1, l[i] - 1):
                for j_start in range(max(0, j_end - self.max_seg_len), j_end):
                    logprob = alpha[j_start] + probs[j_start + 1]\
                              [j_end - j_start - 1][i].item()
                    if logprob > alpha[j_end]:
                        alpha[j_end] = logprob
                        prev[j_end] = j_start
            
            j_end = l[i] - 2
            segment_lengths = []
            while j_end > 0:
                prev_j = prev[j_end]
                segment_lengths.append(j_end - prev_j)
                j_end = prev_j

            segment_lengths = segment_lengths[::-1]

            ret.append(segment_lengths)
        
        segs = [list(np.cumsum(ret[i])) for i in range(batch_size)]
        
        for i in range(batch_size):
            seg_start = 1
            for j, seg_end in enumerate(segs[i]):
                _seg = input[i][seg_start : seg_end + 1]
                _seg = [self.inv_dict[s.item()] for s in _seg]
                segs[i][j] = ''.join(_seg)
                seg_start = seg_end + 1
        return segs, ret