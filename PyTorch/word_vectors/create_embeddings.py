#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:00:32 2021
Load a trained model and extract word vectors.
@author: danny
"""
import torch
import pickle 
import tables
import sys
import torch.nn as nn
import numpy as np

from collections import defaultdict
from torch.utils.data import DataLoader

sys.path.append('../functions')
from minibatchers import CocoDataset, token_pad_fn, CocoSampler

model_loc = './models/caption_model.32'
indices_loc = './models/coco_indices'
feature_loc = '../../../Databases/coco_features.h5'

meta_loc = './'
split_files = {meta_loc + 'SpokenCOCO_train.json': 'train',
               meta_loc + 'SpokenCOCO_val.json': 'val'
               }

# rnn encoder for characters and tokens has an embedding layer followed by
# n rnn layers and an (multi)attention pooling layer
class text_rnn_encoder(nn.Module):
    def __init__(self, config):
        super(text_rnn_encoder, self).__init__()
        embed = config['embed']
        rnn= config['rnn']
        self.max_len = rnn['max_len']
        self.embed = nn.Embedding(num_embeddings = embed['num_chars'], 
                                  embedding_dim = embed['embedding_dim'], 
                                  sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx']
                                  )
        self.RNN = nn.ModuleList()
        for x in range(len(rnn['n_layers'])):
            self.RNN.append(nn.GRU(input_size = rnn['input_size'][x], 
                                   hidden_size = rnn['hidden_size'][x], 
                                   num_layers = rnn['n_layers'][x],
                                   batch_first = rnn['batch_first'],
                                   bidirectional = rnn['bidirectional'], 
                                   dropout = rnn['dropout']
                                   )
                            )
               
    def forward(self, input, l):
        # embedding layers expect Long tensors
        x = self.embed(input.long())
        x = torch.nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True,
                                                    enforce_sorted = False
                                                    )
        x, hx = self.RNN[0](x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)   
        return x

def read_data(h5_file):       
    h5_file = tables.open_file(h5_file, 'r')
    for subgroup in h5_file.root:
        for node in subgroup:
            yield node

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

indices = load_obj(indices_loc)

# create config dictionaries with all the parameters for your encoders
char_config = {'embed':{'num_chars': len(indices), 'embedding_dim': 300, 
                        'sparse': False, 'padding_idx': 0
                        }, 
                'rnn':{'input_size': [300, 2048], 
                       'hidden_size': [1024, 300], 
                       'n_layers': [1], 'batch_first': True, 
                       'bidirectional': True, 'dropout': 0, 
                       'max_len': 128
                       }
                }

cap_net = text_rnn_encoder(char_config)
if torch.cuda.is_available():
    cap_state = torch.load(model_loc, map_location = torch.device('cuda'))
    cap_net.load_state_dict(cap_state, strict = False)
    cap_net.cuda() 
    dtype = torch.cuda.FloatTensor
else:
    cap_state = torch.load(model_loc, map_location = torch.device('cpu'))
    cap_net.load_state_dict(cap_state, strict = False)
    dtype = torch.FloatTensor

cap_net.eval()
for param in cap_net.parameters():
    param.requires_grad = False 
    
dataset = CocoDataset(feature_loc, 'resnet', 'tokens', split_files)
batcher = DataLoader(dataset, batch_size = 1024, 
                     collate_fn = token_pad_fn(256, dtype, 
                                                    'word', indices_loc),
                          sampler = CocoSampler(dataset, 'train', False))

ind_to_word = {indices[key]:key for key in indices.keys()}

word_vecs = {}
for word in indices.keys():
    word_vecs[word] = np.zeros(600)
freq = defaultdict(int)
i = 1
for batch in batcher:
    img, cap, lengths = batch
    cap.requires_grad_(False)
    cap_embedding = cap_net(cap, lengths)
    for x, sent in enumerate(cap_embedding):
        for y, word in enumerate(sent):
            w = ind_to_word[int(cap[x][y].cpu().numpy())]
            word_vecs[w] += word.cpu().data.numpy()
            freq[w] += 1
    print(i)
    i+=1

norm_word_vec = {}

for word in word_vecs.keys():
    vec = word_vecs[word]
    norm = vec / np.sqrt((sum(vec**2)))
    norm_word_vec[word] = norm
        
out = open('my_vecs.txt', 'w')
for x in norm_word_vec.keys():
    vec = norm_word_vec[x]
    formatted = x + ' ' + ' '.join([str(y) for y in vec])
    out.write(formatted + '\n')   
out.close()      
    
