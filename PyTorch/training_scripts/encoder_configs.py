#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:56:17 2020
put the config dictionaries for different types of encoders here
@author: danny
"""

from encoders import (img_encoder, audio_rnn_encoder, conv_VQ_encoder,
                      rnn_pack_encoder, text_rnn_encoder)

def create_encoders(preset_name, dict_size):
    if preset_name == 'rnn':
        # create config dictionaries with all the parameters for your encoders
        audio_config = {'conv':{'in_channels': 39, 'out_channels': 64, 
                                'kernel_size': 6, 'stride': 2,'padding': 0, 
                                'bias': False
                                }, 
                        'rnn':{'input_size': [64], 'hidden_size': [1024], 
                               'n_layers': [4], 'batch_first': True, 
                               'bidirectional': True, 'dropout': 0, 
                               'max_len': 1024
                               }, 

                        'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1
                               },
                        'VQ':{'n_layers': 0, 'n_embs': [], 'emb_dim': []},
                        'app_order': [0]
                        }
        # calculate the required output size of the image encoder
        out_size = audio_config['rnn']['hidden_size'][-1] * 2 ** \
                   audio_config['rnn']['bidirectional'] * audio_config['att']['heads']          
        image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 
                        'norm': True
                        }
        img_net = img_encoder(image_config)
        cap_net = audio_rnn_encoder(audio_config)
        
    elif preset_name == 'rnn_VQ':
        # create config dictionaries with all the parameters for your encoders
        audio_config = {'conv':{'in_channels': 39, 'out_channels': 64, 
                                'kernel_size': 6, 'stride': 2,'padding': 0, 
                                'bias': False
                                }, 
                        'rnn':{'input_size': [64, 2048, 2048], 
                               'hidden_size': [1024, 1024, 1024], 
                               'n_layers': [1, 1, 2], 'batch_first': True, 
                               'bidirectional': True, 'dropout': 0, 
                               'max_len': 1024
                               }, 
                        'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1
                               },
                        'VQ':{'n_layers': 2, 'n_embs': [128, 2048], 
                              'emb_dim': [2048, 2048]
                              },
                        'app_order': [0, 1, 0, 1, 0],
                        }
        # calculate the required output size of the image encoder
        out_size = audio_config['rnn']['hidden_size'][-1] * 2 ** \
                   audio_config['rnn']['bidirectional'] * audio_config['att']['heads']          
        image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 
                        'norm': True
                        }
        img_net = img_encoder(image_config)
        cap_net = audio_rnn_encoder(audio_config)

    elif preset_name == 'rnn_pack':        
        audio_config = {'conv':{'in_channels': 39, 'out_channels': 64, 
                                'kernel_size': 6, 'stride': 2,'padding': 0, 
                                'bias': False
                                }, 
                        'rnn':{'input_size': [64, 1024], 
                               'hidden_size': [1024, 1024], 
                               'n_layers': [1,3], 'batch_first': True, 
                               'bidirectional': [True, True], 'dropout': 0, 
                               'max_len': 1024
                               }, 
                        'rnn_pack':{'input_size': [2048], 'hidden_size': [1024]},
                        'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1
                               },
                        'VQ':{'n_layers': 1, 'n_embs': [64], 
                              'emb_dim': [2048]
                              },
                        'app_order': ['rnn', 'VQ', 'rnn_pack', 'rnn'],
                        }

        out_size = audio_config['rnn']['hidden_size'][-1] * 2 ** \
                   audio_config['rnn']['bidirectional'][-1] * audio_config['att']['heads']          
        image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 
                        'norm': True
                        }
        img_net = img_encoder(image_config)
        cap_net = rnn_pack_encoder(audio_config)
    
           
    elif preset_name == 'conv_VQ':
        
        audio_config = {'conv_init':{'in_channels': 39, 'out_channels': 128, 
                                     'kernel_size': 1, 'stride': 1, 
                                     'padding': 0
                                     },
                        'conv':{'in_channels': [128, 128, 256, 512], 
                                'out_channels': [128, 256, 512, 1024], 
                                'kernel_size': [9, 9, 9, 9], 
                                'stride': [2, 2, 2, 2], 'n_layers': 4 
                                },
                        'att':{'in_size': 1024, 'hidden_size': 128, 
                               'heads': 1
                               },
                        'VQ':{'n_layers': 2, 'n_embs': [1024, 1024], 
                              'emb_dim': [128, 256]
                              },
                        'max_len': 1024, 'app_order': [0, 1, 0, 1, 0, 0]
                        }
        # get the required output size of the img encoder from audio_config
        out_size = audio_config['conv']['out_channels'][-1]          
        image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 
                        'norm': True
                        }
        img_net = img_encoder(image_config)
        cap_net = conv_VQ_encoder(audio_config)
        
    elif preset_name == 'conv':

        audio_config = {'conv_init':{'in_channels': 39, 'out_channels': 128, 
                                     'kernel_size': 1, 'stride': 1, 
                                     'padding': 0
                                     },
                        'conv':{'in_channels': [128, 128, 256, 512], 
                                'out_channels': [128, 256, 512, 1024], 
                                'kernel_size': [9, 9, 9, 9], 
                                'stride': [2, 2, 2, 2], 'n_layers': 4 
                                },
                        'att':{'in_size': 1024, 'hidden_size': 128, 
                               'heads': 1
                               },
                        'VQ':{'n_layers': 0, 'n_embs': [], 'emb_dim': []},
                        'max_len': 1024, 'app_order': [0, 0, 0, 0]
                        }
        # get the required output size of the img encoder from audio_config
        out_size = audio_config['conv']['out_channels'][-1]          
        image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 
                        'norm': True
                        }
        img_net = img_encoder(image_config)
        cap_net = conv_VQ_encoder(audio_config)

    elif preset_name == 'rnn_text':
        # create config dictionaries with all the parameters for your encoders
        char_config = {'embed':{'num_chars': dict_size, 'embedding_dim': 1024, 
                                'sparse': False, 'padding_idx': 0
                                }, 
                        'rnn':{'input_size': 1024, 
                               'hidden_size': 1024, 
                               'n_layers': 1, 'batch_first': True, 
                               'bidirectional': True, 'dropout': 0, 
                               'max_len': 1024
                               }, 
                        'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1
                               }
                        }
        # calculate the required output size of the image encoder
        out_size = char_config['rnn']['hidden_size'] * 2 ** \
                   char_config['rnn']['bidirectional'] * char_config['att']['heads']          
        image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 
                        'norm': True
                        }
        img_net = img_encoder(image_config)
        cap_net = text_rnn_encoder(char_config)
        
    return(img_net, cap_net)