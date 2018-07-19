#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:13:00 2018

@author: danny
"""
#!/usr/bin/env python
from __future__ import print_function

import time
import tables
import argparse
import torch
from torch.autograd import Variable
import numpy as np
from torch.optim import lr_scheduler
import sys
sys.path.append('/data/speech2image/PyTorch/functions')

from minibatchers import iterate_audio_5fold, iterate_audio
from costum_loss import batch_hinge_loss, ordered_loss
from evaluate import caption2image, image2caption
from encoders import img_encoder, audio_gru_encoder
from data_split import split_data
from grad_tracker import gradient_clipping
##################################### parameter settings ##############################################

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, default = '/prep_data/flickr_features.h5',
                    help = 'location of the feature file, default: /prep_data/flickr_features.h5')
parser.add_argument('-split_loc', type = str, default = '/data/speech2image/preprocessing/dataset.json', 
                    help = 'location of the json file containing the data split information')
parser.add_argument('-results_loc', type = str, default = '/data/speech2image/PyTorch/flickr_audio/results/',
                    help = 'location to save the trained models')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 32, help = 'batch size, default: 32')
parser.add_argument('-lr', type = float, default = 0.0002, help = 'learning rate, default:0.0002')
parser.add_argument('-n_epochs', type = int, default = 32, help = 'number of training epochs, default: 25')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
# args concerning the database and which features to load
parser.add_argument('-data_base', type = str, default = 'flickr', help = 'database to train on, default: flickr')
parser.add_argument('-visual', type = str, default = 'resnet', help = 'name of the node containing the visual features, default: resnet')
parser.add_argument('-cap', type = str, default = 'mfcc', help = 'name of the node containing the audio features, default: mfcc')
parser.add_argument('-gradient_clipping', type = bool, default = False, help ='use gradient clipping, default: False')

args = parser.parse_args()

# create config dictionaries with all the parameters for your encoders

audio_config = {'conv':{'in_channels': 39, 'out_channels': 64, 'kernel_size': 6, 'stride': 2,
               'padding': 0, 'bias': False}, 'gru':{'input_size': 64, 'hidden_size': 1024, 
               'num_layers': 4, 'batch_first': True, 'bidirectional': True, 'dropout': 0}, 
               'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}}
# automatically adapt the image encoder output size to the size of the caption encoder
out_size = audio_config['gru']['hidden_size'] * 2**audio_config['gru']['bidirectional']
image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 'norm': True}


# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 

# check if cuda is availlable and user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
    # if cuda cast all variables as cuda tensor
    dtype = torch.cuda.FloatTensor
else:
    print('using cpu')
    dtype = torch.FloatTensor

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
    # define the batcher type to use.
    batcher = iterate_audio_5fold    
elif args.data_base == 'flickr':
    f_nodes = [node for node in iterate_flickr(data_file)]
    # define the batcher type to use.
    batcher = iterate_audio_5fold
elif args.data_base == 'places':
    f_nodes = [node for node in iterate_large_dataset(data_file)]
    batcher = iterate_audio
else:
    print('incorrect database option')
    exit()  

# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
train, test, val = split_data(f_nodes, args.split_loc)

############################### Neural network setup #################################################

# network modules
img_net = img_encoder(image_config)
cap_net = audio_gru_encoder(audio_config)

if args.gradient_clipping:
    img_clipper = gradient_clipping(clip_value = 0.0015)
    cap_clipper = gradient_clipping(clip_value = 0.015)
    
    img_clipper.register_hook(img_net)
    cap_clipper.register_hook(cap_net)
    
# move graph to gpu if cuda is availlable
if cuda:
    img_net.cuda()
    cap_net.cuda()
# function to save parameters in a results folder
def save_params(model, file_name, epoch):
    torch.save(model.state_dict(), args.results_loc + file_name + '.' +str(epoch))

# Adam optimiser. I found SGD to work terribly and could not find appropriate parameter settings for it.
optimizer = torch.optim.Adam(list(img_net.parameters())+list(cap_net.parameters()), 1)

def create_cyclic_scheduler(max_lr, min_lr, stepsize):
    # lambda function which uses the cosine function to cycle the learning rate between the given min and max rates
    # the function operates between 1 and 3 (so the cos cycles from -1 to -1 ) normalise between 0 and 1 and then press between
    # min and max lr 
    lr_lambda = lambda iteration: (max_lr - min_lr)*(0.5 * (np.cos(np.pi * (1 + (3 - 1) / stepsize * iteration)) + 1))+min_lr
    cyclic_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)  
    return(cyclic_scheduler)

cyclic_scheduler = create_cyclic_scheduler(max_lr = args.lr, min_lr = 1e-6, stepsize = (int(len(train)/args.batch_size)*5)*4)

# training routine 
def train_epoch(epoch, img_net, cap_net, optimizer, f_nodes, batch_size):
    global iteration
    img_net.train()
    cap_net.train()
    # for keeping track of the average loss over all batches
    train_loss = 0
    num_batches =0
    for batch in batcher(f_nodes, batch_size, args.visual, args.cap, frames = 2048, shuffle = True):
        cyclic_scheduler.step()
        iteration +=1 
        img, cap, lengths = batch
        num_batches +=1
        # sort the tensors based on the unpadded caption length so they can be used
        # with the pack_padded_sequence function
        cap = cap[np.argsort(- np.array(lengths))]
        img = img[np.argsort(- np.array(lengths))]
        lengths = np.array(lengths)[np.argsort(- np.array(lengths))]
        print(lengths)
        # convert data to pytorch variables
        img, cap = Variable(dtype(img)), Variable(dtype(cap))
        # reset the gradients of the optimiser
        optimizer.zero_grad()
        # embed the images and audio using the networks
        img_embedding = img_net(img)
        cap_embedding = cap_net(cap, lengths)
        # calculate the loss
        loss = batch_hinge_loss(img_embedding, cap_embedding, cuda)
        # calculate the gradients and perform the backprop step
        loss.backward()
        if args.gradient_clipping:
            torch.nn.utils.clip_grad_norm(img_net.parameters(), img_clipper.clip)
            torch.nn.utils.clip_grad_norm(cap_net.parameters(), cap_clipper.clip)
        optimizer.step()
        # add loss to average
        train_loss += loss.data
        print(train_loss.cpu()[0]/num_batches)
    return train_loss/num_batches

def test_epoch(img_net, cap_net, f_nodes, batch_size):
    # set to evaluation mode
    img_net.eval()
    cap_net.eval()
    # for keeping track of the average loss
    test_batches = 0
    test_loss = 0
    for batch in batcher(f_nodes, batch_size, args.visual, args.cap, frames = 2048, shuffle = False):
        img, cap, lengths = batch 
        test_batches += 1
        # sort the tensors based on the unpadded caption length so they can be used
        # with the pack_padded_sequence function
        cap = cap[np.argsort(- np.array(lengths))]
        img = img[np.argsort(- np.array(lengths))]
        lengths = np.array(lengths)[np.argsort(- np.array(lengths))]
        
        # convert data to pytorch variables
        img, cap = Variable(dtype(img)), Variable(dtype(cap))
        # embed the images and audio using the networks
        img_embedding = img_net(img)
        cap_embedding = cap_net(cap, lengths)
        loss = batch_hinge_loss(img_embedding, cap_embedding, cuda)
        # add loss to average
        test_loss += loss.data 
    return test_loss/test_batches 

def report(start_time, train_loss, val_loss, epoch):
    # report on the time and train and val loss for the epoch
    print("Epoch {} of {} took {:.3f}s".format(
            epoch, args.n_epochs, time.time() - start_time))
    print("training loss:\t\t{:.6f}".format(train_loss.cpu()[0]))
    print("validation loss:\t\t{:.6f}".format(val_loss.cpu()[0]))
    
def recall(data, at_n, c2i, i2c, prepend):
    # calculate the recall@n. Arguments are a set of nodes, the @n values, whether to do caption2image, image2caption or both
    # and a prepend string (e.g. to print validation or test in front of the results)
    if c2i:
        # create a minibatcher over the validation set
        iterator = batcher(data, args.batch_size, args.visual, args.cap, frames = 2048, shuffle = False)
        recall, median_rank = caption2image(iterator, img_net, cap_net, at_n, dtype)
        # print some info about this epoch
        for x in range(len(recall)):
            print(prepend + ' caption2image recall@' + str(at_n[x]) + ' = ' + str(recall[x]*100) + '%')
        print(prepend + ' caption2image median rank= ' + str(median_rank))
    if i2c:
        # create a minibatcher over the validation set
        iterator = batcher(data, args.batch_size, args.visual, args.cap, frames = 2048, shuffle = False)
        recall, median_rank = image2caption(iterator, img_net, cap_net, at_n, dtype)
        for x in range(len(recall)):
            print(prepend + ' image2caption recall@' + str(at_n[x]) + ' = ' + str(recall[x]*100) + '%')
        print(prepend + ' image2caption median rank= ' + str(median_rank))    

################################# training/test loop #####################################
epoch = 1
iteration = 0 
# run the training loop for the indicated amount of epochs 
while epoch <= args.n_epochs:
    # keep track of runtime
    start_time = time.time()

    print('training epoch: ' + str(epoch))
    # Train on the train set
    train_loss = train_epoch(epoch, img_net, cap_net, optimizer, train, args.batch_size)
    
    #evaluate on the validation set
    val_loss = test_epoch(img_net, cap_net, val, args.batch_size)
    # save network parameters
    save_params(img_net, 'image_model', epoch)
    save_params(cap_net, 'caption_model', epoch)
    
    # print some info about this epoch
    report(start_time, train_loss, val_loss, epoch)
    recall(val, [1, 5, 10], c2i = True, i2c = False, prepend = 'validation')    
    epoch += 1
    # this part is usefull only if you want to update the value for gradient clipping at each epoch
    # I found it didn't work well 
    #if args.gradient_clipping:
        #cap_clipper.update_clip_value()
        #cap_clipper.reset_gradients()
        #img_clipper.update_clip_value()
        #img_clipper.reset_gradients()
    
test_loss = test_epoch(img_net, cap_net, test, args.batch_size)
print("test loss:\t\t{:.6f}".format(test_loss.cpu()[0]))# calculate the recall@n
recall(test, [1, 5, 10], c2i = True, i2c = True, prepend = 'test')

# save the gradients for each epoch, can be usefull to select an initial clipping value.
if args.gradient_clipping:
    cap_clipper.save_grads(args.results_loc, 'textgrads')
    img_clipper.save_grads(args.results_loc, 'imgrads')

