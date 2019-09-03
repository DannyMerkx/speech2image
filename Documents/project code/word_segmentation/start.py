#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:52:39 2019

@author: danny

"""
import numpy as np
import torch 
import sys

sys.path.append('/home/danny/Documents/project code/word_segmentation')

# from costum_loss import likelihood_loss
from likelihood_loss import likelihood_loss
from SNLM_trainer import SNLM_trainer
from collections import defaultdict
from torch.optim import lr_scheduler

########################### data preparation ##################################

raw_data = [x for x in open('/home/danny/Downloads/brent_ratner/br-phono.txt')]
raw_data = list(set(raw_data))

#np.random.shuffle(raw_data)

# function to prepare the text data, remove all whitespace and then separate
# all characters by spaces and add in an end of sentence token
def prep_data(data):
    
    white_space = [' ', '\n', '\t']
    
    for idx, sent in enumerate(data):
        for ws in white_space:
            data[idx] = data[idx].replace(ws, '')
            
        data[idx] = '<w> ' + ' '.join(data[idx]) + ' </w>'   
        
    return data
# create the dictionary which maps all tokens in the dataset to an embedding
# index
def create_token_dict(data):
    token_dict = defaultdict(int)
    
    for sent in data:
        for token in sent.split(' '):
            if token_dict[token] == 0:
                token_dict[token] = len(token_dict)
                
    # add end and beginning of word tokens. 
    #token_dict['</w>'] = len(token_dict) + 1
    #token_dict['<w>'] = len(token_dict) + 1
    
    return token_dict

data = prep_data(list(raw_data))

train = data[0 : int(len(data)*.8)]
gs_train = raw_data[0 : int(len(data)*.8)]
val = data[int(len(data)*.8) : int(len(data) * .8 + len(data) * .1)]
gs_val = raw_data[int(len(data)*.8) : int(len(data) * .8 + len(data) * .1)]
test = data[int(len(data) * .8 + len(data) * .1) :]
gs_test = raw_data[int(len(data) * .8 + len(data) * .1) :]

token_dict = create_token_dict(data)

###############################################################################

# configuration settings for the neural networks. The required linear layer 
# size is the hidden size of the rnn and the number of embeddings
hist_config = {'embed': {'n_embeddings': len(token_dict) + 1, 
                         'embedding_dim': 512, 'sparse': False, 
                         'padding_idx': 0
                         }, 
               'rnn': {'input_size': 512, 'hidden_size': 512, 
                       'num_layers': 1, 'batch_first': False, 
                       'bidirectional': False, 'dropout': 0
                       }
               }

char_config = {'embed': {'n_embeddings': len(token_dict) + 1, 
                         'embedding_dim': 512, 'sparse': False, 
                         'padding_idx': 0
                        }, 
              'rnn': {'input_size': 512, 'hidden_size': 512, 
                      'num_layers': 2, 'batch_first': False, 
                      'bidirectional': False, 'dropout': 0
                      },
              'hist_size': hist_config['rnn']['hidden_size']
              }

tf_config = {'embed': {'n_embeddings': len(token_dict) + 1, 
                        'embedding_dim': 128, 'sparse': False, 'padding_idx': 0
                       }, 
              'tf': {'input_size': 128, 'fc_size': 512, 
                      'n_layers': 1, 'h': 8, 
                      'max_len': 74
                      },
              'cuda': False
              }

trainer = SNLM_trainer(char_config, hist_config, token_dict) 

loss_function = likelihood_loss(token_dict, max_seg_len = 10)                    
optimizer = torch.optim.Adam(list(trainer.char_encoder.parameters()) +
                             list(trainer.hist_encoder.parameters()), 
                             lr = 0.01)
trainer.set_optimizer(optimizer)
trainer.set_loss(loss_function)
trainer.set_grad_clipping(1)
def create_cyclic_scheduler(max_lr, min_lr, stepsize):
    lr_lambda = lambda iteration: (max_lr - min_lr) * (0.5 * (np.cos(np.pi * \
                                  (1 + (3 - 1) / stepsize * iteration)) + 1))\
                                  + min_lr
    
    cyclic_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, 
                                             last_epoch = -1)
    return(cyclic_scheduler)

#cyclic_scheduler = create_cyclic_scheduler(max_lr = 0.01, min_lr = 1e-6, 
#                                           stepsize = int(len(data)/110)*4)

n_epochs = 4

while trainer.epoch <= n_epochs:
    trainer.train_loop(train, 110, max([len(x.split()) for x in train]), val, gs_val)
    x = trainer.test_loop(val, gs_val, max([len(x.split()) for x in val]))
    trainer.update_epoch()
    
