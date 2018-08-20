#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:13:00 2018

@author: danny
"""
#!/usr/bin/env python
from __future__ import print_function

import tables
import argparse
import torch
import numpy as np
from torch.optim import lr_scheduler
import sys
sys.path.append('/data/speech2image/PyTorch/functions')

from trainer import flickr_trainer
from costum_loss import batch_hinge_loss, ordered_loss, attention_loss
from encoders import img_encoder, char_gru_encoder
from data_split import split_data_coco
##################################### parameter settings ##############################################

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, default = '/prep_data/coco_features.h5',
                    help = 'location of the feature file, default: /prep_data/coco_features.h5')
parser.add_argument('-results_loc', type = str, default = '/data/speech2image/PyTorch/coco_words/results/',
                    help = 'location to save the results and network parameters')
parser.add_argument('-dict_loc', type = str, default = '/data/speech2image/preprocessing/dictionaries/coco_indices')
parser.add_argument('-glove_loc', type = str, default = '/data/glove.840B.300d.txt', help = 'location of pretrained glove embeddings')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 32, help = 'batch size, default: 32')
parser.add_argument('-lr', type = float, default = 0.0001, help = 'learning rate, default:0.0001')
parser.add_argument('-n_epochs', type = int, default = 32, help = 'number of training epochs, default: 25')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
parser.add_argument('-glove', type = bool, default = False, help = 'use pretrained glove embeddings, default: False')
# args concerning the database and which features to load
parser.add_argument('-data_base', type = str, default = 'coco', help = 'database to train on, default: coco')
parser.add_argument('-visual', type = str, default = 'resnet', help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'cannonical_tokens', help = 'name of the node containing the caption features, default: tokens')
parser.add_argument('-gradient_clipping', type = bool, default = False, help ='use gradient clipping, default: False')

args = parser.parse_args()

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
# get the size of the dictionary for the embedding layer (pytorch crashes if the embedding layer is not correct for the dictionary size)
# add 1 for the zero or padding embedding
dict_size = len(load_obj(args.dict_loc))

# create config dictionaries with all the parameters for your encoders
char_config = {'embed':{'num_chars': dict_size, 'embedding_dim': 300, 'sparse': False, 'padding_idx': 0}, 
               'gru':{'input_size': 300, 'hidden_size': 1024, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}}
# automatically adapt the image encoder output size to the size of the caption encoder
out_size = char_config['gru']['hidden_size'] * 2**char_config['gru']['bidirectional']
image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 'norm': True}

# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 

# check if cuda is availlable and if user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
else:
    print('using cpu')

# get a list of all the nodes in the file. h5 format takes at most 10000 leaves per node, so big
# datasets are split into subgroups at the root node 
def iterate_large_dataset(h5_file):
    for x in h5_file.root:
        for y in x:
            yield y
# flickr doesnt need to be split at the root node
def iterate_flickr(h5_file):
    for x in h5_file.root:
        yield x

if args.data_base == 'coco':
    f_nodes = [node for node in iterate_large_dataset(data_file)]   
elif args.data_base == 'flickr':
    f_nodes = [node for node in iterate_flickr(data_file)]
elif args.data_base == 'places':
    print('places has no written captions')
else:
    print('incorrect database option')
    exit()  

# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
train, val = split_data_coco(f_nodes)
# set aside 5000 images as test set
test = train[-5000:]
train = train[:-5000]

############################### Neural network setup #################################################
# network modules
img_net = img_encoder(image_config)
cap_net = char_gru_encoder(char_config)
    
# Adam optimiser. I found SGD to work terribly and could not find appropriate parameter settings for it.
optimizer = torch.optim.Adam(list(img_net.parameters())+list(cap_net.parameters()), 1)

#plateau_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.9, patience = 100, 
#                                                   threshold = 0.0001, min_lr = 1e-8, cooldown = 100)

#step_scheduler = lr_scheduler.StepLR(optimizer, 1000, gamma=0.1, last_epoch=-1)

def create_cyclic_scheduler(max_lr, min_lr, stepsize):
    lr_lambda = lambda iteration: (max_lr - min_lr)*(0.5 * (np.cos(np.pi * (1 + (3 - 1) / stepsize * iteration)) + 1))+min_lr
    cyclic_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    # lambda function which uses the cosine function to cycle the learning rate between the given min and max rates
    # the function operates between 1 and 3 (so the cos cycles from -1 to -1 ) normalise between 0 and 1 and then press between
    # min and max lr   
    return(cyclic_scheduler)

cyclic_scheduler = create_cyclic_scheduler(max_lr = args.lr, min_lr = 1e-6, stepsize = (int(len(train)/args.batch_size)*5)*4)

# create a trainer setting the loss function, optimizer, minibatcher, lr_scheduler and the r@n evaluator
trainer = flickr_trainer(img_net, cap_net, args.visual, args.cap)
trainer.set_loss(batch_hinge_loss)
trainer.set_optimizer(optimizer)
trainer.set_token_batcher()
trainer.set_dict_loc(args.dict_loc)
trainer.set_lr_scheduler(cyclic_scheduler, 'cyclic')
# optionally use cuda, gradient clipping and pretrained glove vectors
if cuda:
    trainer.set_cuda()
# load pretrained glove vectors and freeze the embedding layer
if args.glove:
    trainer.load_glove_embeddings(args.glove_loc)
    trainer.cap_embedder.embed.weight.requires_grad = False
    parameters = filter(lambda p: p.requires_grad, trainer.cap_embedder.parameters())
    optimizer = torch.optim.Adam(list(img_net.parameters())+list(parameters), 1)
# gradient clipping with these parameters (based the avg gradient norm for the first epoch)
# can help stabilise training in the first epoch.
if args.gradient_clipping:
    trainer.set_gradient_clipping(0.0025, 0.05)
    
################################# training/test loop #####################################

# run the training loop for the indicated amount of epochs 
while trainer.epoch <= args.n_epochs:
    # Train on the train set
    trainer.train_epoch(train, args.batch_size)
    #evaluate on the validation set
    trainer.test_epoch(val, args.batch_size)
    # save network parameters
    trainer.save_params(args.results_loc)  
    # print some info about this epoch
    trainer.report(args.n_epochs)
    trainer.recall_at_n(val, args.batch_size, prepend = 'validation')    
    trainer.fivefold_recall_at_n('validation')
    if args.gradient_clipping:
        # I found that updating the clip value at each epoch did not work well     
        # trainer.update_clip()
        trainer.reset_grads()
    #increase epoch#
    trainer.update_epoch()
trainer.test_epoch(test, args.batch_size)
trainer.print_test_loss()
# calculate the recall@n
trainer.recall_at_n(test, args.batch_size, prepend = 'test')
trainer.fivefold_recall_at_n('test')
# save the gradients for each epoch, can be usefull to select an initial clipping value.
if args.gradient_clipping:
    trainer.save_gradients(args.results_loc)