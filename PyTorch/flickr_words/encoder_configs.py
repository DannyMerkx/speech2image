#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:56:17 2020
put the config dictionaries for different types of encoders here
@author: danny
"""
import pickle
from encoders import (img_encoder, text_rnn_encoder, c2i_transformer)

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

def create_encoders(preset_name, dict_loc, cuda):
    if preset_name == 'rnn':
        # get the size of the dictionary for the embedding layer (pytorch crashes if the embedding layer is not correct for the dictionary size)
        # add 1 for the zero or padding embedding
        dict_size = len(load_obj(dict_loc))
        # create config dictionaries with all the parameters for your encoders
        token_config = {'embed':{'num_chars': dict_size, 'embedding_dim': 300, 
                                 'sparse': False, 'padding_idx': 0
                                 }, 
                       'rnn':{'input_size': 300, 'hidden_size': 1024, 
                              'num_layers': 1,  'batch_first': True, 
                              'bidirectional': True, 'dropout': 0, 
                              'max_len': 64
                              }, 
                       'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}
                       }
        # automatically adapt the image encoder output size to the size of the caption encoder
        out_size = token_config['rnn']['hidden_size'] * 2 ** \
                   token_config['rnn']['bidirectional'] * token_config['att']['heads']
        image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 
                        'norm': True
                        }
        img_net = img_encoder(image_config)
        cap_net = text_rnn_encoder(token_config)
        
    elif preset_name == 'transformer':
        
        dict_size = len(load_obj(dict_loc))
        token_config = {'embed': {'n_embeddings': dict_size,
                                  'embedding_dim': 400, 'sparse': False, 
                                  'padding_idx':0
                                  }, 
                        'tf':{'in_size':400, 'fc_size': 1024,'n_layers': 6,
                              'h': 8, 'max_len': 64
                              },  
                        'cuda': cuda
                        }
        # automatically adapt the image encoder output size to the size of the caption encoder
        out_size = token_config['rnn']['hidden_size'] * 2 ** \
                   token_config['rnn']['bidirectional'] * token_config['att']['heads']
        image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 
                        'norm': True
                        }
        img_net = img_encoder(image_config)
        cap_net = text_rnn_encoder(token_config)
        
    return(img_net, cap_net)

