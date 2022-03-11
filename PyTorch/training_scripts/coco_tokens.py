#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:13:00 2018

@author: danny
"""
#!/usr/bin/env python
from __future__ import print_function

import argparse
import torch
import pickle
from torch.optim import lr_scheduler
import sys
sys.path.append('../functions')

from trainer import flickr_trainer
from costum_loss import batch_hinge_loss, ordered_loss, attention_loss
from costum_scheduler import cyclic_scheduler
from minibatchers import CocoDataset
from encoder_configs import create_encoders
##################################### parameter settings ##############################################

parser = argparse.ArgumentParser(description = 
                                 'Create and run an articulatory feature classification DNN')
# args concerning file location
parser.add_argument('-data_loc', type = str, 
                    default = '//Data/coco_features.h5',
                    help = 'location of the feature file')
parser.add_argument('-results_loc', type = str, 
                    default = '/vol/tensuser3/dmerkx/results/',
                    help = 'location to save the results and network parameters')
parser.add_argument('-dict_loc', type = str, 
                    default = '/vol/tensuser3/dmerkx/databases/flickr8k/flickr_indices')
parser.add_argument('-glove_loc', type = str, 
                    default = '/vol/tensuser3/dmerkx/databases/glove.840B.300d.txt', 
                    help = 'location of pretrained glove embeddings')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 32, 
                    help = 'batch size, default: 32')
parser.add_argument('-lr', type = float, default = 0.001, 
                    help = 'learning rate, default:0.001')
parser.add_argument('-n_epochs', type = int, default = 32, 
                    help = 'number of training epochs, default: 32')
parser.add_argument('-cuda', type = bool, default = True, 
                    help = 'use cuda, default: True')
parser.add_argument('-glove', type = bool, default = False, 
                    help = 'use pretrained glove embeddings, default: False')
# args concerning the database and which features to load
parser.add_argument('-visual', type = str, default = 'resnet', 
                    help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'tokens', 
                    help = 'name of the node containing the caption features, default: tokens')
parser.add_argument('-gradient_clipping', type = bool, default = False, 
                    help ='use gradient clipping, default: False')

args = parser.parse_args()

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
dict_size = len(load_obj(args.dict_loc))

#use the coco split by Harwath et al. so we can compare to SpokenCOCO models
meta_loc = '/vol/tensusers2/dmerkx/coco/SpokenCOCO/'
split_files = {meta_loc + 'SpokenCOCO_train.json': 'train',
               meta_loc + 'SpokenCOCO_val.json': 'val'
               }

# create encoders using presets defined in encoder_configs
img_net, cap_net = create_encoders('rnn_text', dict_size)

# open the dataset
dataset = CocoDataset(args.data_loc, args.visual, args.cap, split_files)

# check if cuda is availlable and if user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
else:
    print('using cpu')
############################### Neural network setup #################################################
# Adam optimiser. I found SGD to work terribly and could not find appropriate parameter settings for it.
optimizer = torch.optim.Adam(list(img_net.parameters())+
                             list(cap_net.parameters()), 1)

# plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', 
#                                                    factor = 0.9, 
#                                                    patience = 100, 
#                                                    threshold = 0.0001, 
#                                                    min_lr = 1e-8, 
#                                                    cooldown = 100)

#step_scheduler = lr_scheduler.StepLR(optimizer, 1000, gamma=0.1, last_epoch=-1)
cyclic_scheduler = cyclic_scheduler(max_lr = args.lr, min_lr = 1e-6, 
                                    stepsize = (int(len(dataset.train)/args.batch_size)*5)*4,
                                    optimiser = optimizer)

# create a trainer setting the loss function, optimizer, minibatcher, lr_scheduler and the r@n evaluator
trainer = flickr_trainer(img_net, cap_net, args.visual, args.cap)
trainer.set_loss(batch_hinge_loss)
trainer.set_optimizer(optimizer)
trainer.set_token_batcher(args.dict_loc)
trainer.set_dict_loc(args.dict_loc)
trainer.set_lr_scheduler(cyclic_scheduler, 'cyclic')
trainer.set_att_loss(attention_loss)
# optionally use cuda, gradient clipping and pretrained glove vectors
if cuda:
    trainer.set_cuda()
# load pretrained glove vectors and freeze the embedding layer
if args.glove:
    # if we load glove vectors we need to freeze the embedding layer and reset the 
    # optimizer and lr scheduler
    trainer.load_glove_embeddings(args.glove_loc)
    trainer.cap_embedder.embed.weight.requires_grad = False
    parameters = filter(lambda p: p.requires_grad, trainer.cap_embedder.parameters())
    optimizer = torch.optim.Adam(list(img_net.parameters())+list(parameters), 1)
    cyclic_scheduler = cyclic_scheduler(max_lr = args.lr, min_lr = 1e-6, 
                                        stepsize = (int(len(dataset.train)/args.batch_size)*5)*4,
                                        optimiser = optimizer)
    trainer.set_lr_scheduler(cyclic_scheduler, 'cyclic')
    trainer.set_optimizer(optimizer)
    
trainer.set_evaluator([1, 5, 10])
# gradient clipping with these parameters (based the avg gradient norm for the first epoch)
# can help stabilise training in the first epoch.
if args.gradient_clipping:
    trainer.set_gradient_clipping(0.0025, 0.05)
################################# training/test loop #####################################
# run the training loop for the indicated amount of epochs 
while trainer.epoch <= args.n_epochs:
    # Train on the train set
    trainer.train_epoch(dataset, args.batch_size)
    # save network parameters
    trainer.save_params(args.results_loc)  
    # print some info about this epoch and evaluation on the validation set
    trainer.report_training(args.n_epochs, dataset)
    
    if args.gradient_clipping:
        # I found that updating the clip value at each epoch did not work well     
        # trainer.update_clip()
        trainer.reset_grads()
    trainer.update_epoch()
trainer.report_test(dataset)

# save the gradients for each epoch, can be usefull to select an initial clipping value.
if args.gradient_clipping:
    trainer.save_gradients(args.results_loc)