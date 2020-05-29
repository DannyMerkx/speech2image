#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:56:17 2020

@author: danny
"""
from encoders import (img_encoder, audio_rnn_encoder, audio_conv_encoder, 
                      quantized_encoder, conv_VQ_encoder)

def create_encoders(preset_name):
    if preset_name == 'rnn':
        # create config dictionaries with all the parameters for your encoders
        audio_config = {'conv':{'in_channels': 39, 'out_channels': 64, 
                                'kernel_size': 6, 'stride': 2,'padding': 0, 
                                'bias': False
                                }, 
                        'rnn':{'input_size': 64, 'hidden_size': 1024, 'num_layers': 4, 
                               'batch_first': True, 'bidirectional': True, 
                               'dropout': 0, 'max_len':512
                               }, 
                       'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}
                       }
        out_size = audio_config['rnn']['hidden_size'] * 2 ** \
                   audio_config['rnn']['bidirectional'] * audio_config['att']['heads']          
        image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 
                        'norm': True
                        }
        img_net = img_encoder(image_config)
        cap_net = audio_rnn_encoder(audio_config)
        
    elif preset_name == 'conv':

        audio_config = {'conv_init':{'in_channels': 39, 'out_channels': 128, 
                                     'kernel_size': 1, 'stride': 1, 'padding': 0,
                                     },
                        'conv':{'in_channels': [128, 128, 256, 512], 
                                'out_channels': [128, 256, 512, 1024], 
                                'kernel_size': [9, 9, 9, 9], 'stride': [2, 2, 2, 2]
                                },
                        'att':{'in_size': 1024, 'hidden_size': 128, 'heads': 1},
                        'max_len': 1024
                        }
        out_size = 1024           
        image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 
                        'norm': True
                        }
        img_net = img_encoder(image_config)
        cap_net = conv_VQ_encoder(audio_config)
        
    return(img_net, cap_net)
