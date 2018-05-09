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

from minibatchers import iter_text_flickr, iterate_minibatches
from costum_loss import batch_hinge_loss, ordered_loss
from evaluate import speech2image
from encoders import img_encoder, char_gru_encoder
from data_split import split_data
from grad_tracker import gradient_clipping
##################################### parameter settings ##############################################

parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')

# args concerning file location
parser.add_argument('-data_loc', type = str, default = '/prep_data/flickr_features.h5',
                    help = 'location of the feature file, default: /prep_data/flickr_features.h5')
parser.add_argument('-split_loc', type = str, default = '/data/speech2image/PyTorch/flickr_char/dataset.json', 
                    help = 'location of the json file containing the data split information')
parser.add_argument('-results_loc', type = str, default = '/data/speech2image/PyTorch/flickr_char/results/',
                    help = 'location to save the results and network parameters')
# args concerning training settings
parser.add_argument('-batch_size', type = int, default = 32, help = 'batch size, default: 32')
parser.add_argument('-lr', type = float, default = 0.0002, help = 'learning rate, default:0.0002')
parser.add_argument('-n_epochs', type = int, default = 25, help = 'number of training epochs, default: 25')
parser.add_argument('-cuda', type = bool, default = True, help = 'use cuda, default: True')
# args concerning the database and which features to load
parser.add_argument('-data_base', type = str, default = 'flickr', help = 'database to train on, options: places, flickr')
parser.add_argument('-visual', type = str, default = 'vgg19', help = 'name of the node containing the visual features')
parser.add_argument('-audio', type = str, default = 'raw_text', help = 'name of the node containing the audio features')
parser.add_argument('-gradient_clipping', type = bool, default = True, help ='use gradient clipping, default: True')

args = parser.parse_args()

# create config dictionaries with all the parameters for your encoders

char_config = {'embed':{'num_chars': 100, 'embedding_dim': 20, 'sparse': False, 'padding_idx': 0}, 
               'gru':{'input_size': 20, 'hidden_size': 1024, 'num_layers': 4, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 2048, 'hidden_size': 128}}

image_config = {'linear':{'in_size': 4096, 'out_size': 2048}, 'norm': True}

# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 

# check if cuda is availlable and if user wants to run on gpu
cuda = args.cuda and torch.cuda.is_available()
if cuda:
    print('using gpu')
    # if cuda cast all variables as cuda tensor
    dtype = torch.cuda.FloatTensor
else:
    print('using cpu')
    dtype = torch.FloatTensor

# get a list of all the nodes in the file. The places database was so big 
# it needs to be split into subgroups at the root node so the iterator needs
# to look one node deeper in the tree.
def iterate_places(h5_file):
    for x in h5_file.root:
        for y in x:
            yield y
def iterate_flickr(h5_file):
    for x in h5_file.root:
        yield x

if args.data_base == 'places':
    f_nodes = [node for node in iterate_places(data_file)]
    # define the batcher type to use.
    batcher = iterate_minibatches
elif args.data_base == 'flickr':
    f_nodes = [node for node in iterate_flickr(data_file)]
    # define the batcher type to use.
    batcher = iter_text_flickr
else:
    print('incorrect database option')
    exit()  

# split the database into train test and validation sets. default settings uses the json file
# with the karpathy split
train, test, val = split_data(f_nodes, args.split_loc)

############################### Neural network setup #################################################

# network modules
img_net = img_encoder(image_config)
text_net = char_gru_encoder(char_config)

im_params = torch.load(args.results_loc + 'img_model.2')
text_params = torch.load(args.results_loc + 'text_model.2')
img_net.load_state_dict(im_params)
text_net.load_state_dict(text_params)
if args.gradient_clipping:
    img_clipper = gradient_clipping(clip_value = 0.0005)
    text_clipper = gradient_clipping(clip_value = 0.0005)
  
    img_clipper.register_hook(img_net)
    text_clipper.register_hook(text_net)
    
    
# move graph to gpu if cuda is availlable
if cuda:
    img_net.cuda()
    text_net.cuda()
# function to save parameters in a results folder
def save_params(model, file_name, epoch):
    torch.save(model.state_dict(), args.results_loc + file_name + '.' +str(epoch))

# optimiser
optimizer = torch.optim.SGD(list(img_net.parameters())+list(text_net.parameters()), args.lr, momentum =.9)

# this sets the learning rate for the model to a learning rate adapted for the number of
# epochs.
def lr_decay(optimizer, epoch):
    lr = args.lr * (0.2 ** (epoch // 1))
    for groups in optimizer.param_groups:
        groups['lr'] = lr

# training routine 
def train_epoch(epoch, img_net, text_net, optimizer, f_nodes, batch_size):
    img_net.train()
    text_net.train()
    # perform learning rate decay
    lr_decay(optimizer, epoch)
    # for keeping track of the average loss over all batches
    train_loss = 0
    num_batches =0
    for batch in batcher(f_nodes, batch_size, args.visual, args.audio, shuffle = True):
        img, text = batch
        num_batches +=1
        # convert data to pytorch variables
        img, text = Variable(dtype(img)), Variable(dtype(text))
        # reset the gradients of the optimiser
        optimizer.zero_grad()
        # embed the images and audio using the networks
        img_embedding = img_net(img)
        text_embedding = text_net(text)
        # calculate the loss
        loss = batch_hinge_loss(img_embedding, text_embedding, cuda)
        # calculate the gradients and perform the backprop step
        loss.backward()
        # clip the gradients if required
        if args.gradient_clipping:
            torch.nn.utils.clip_grad_norm(img_net.parameters(), img_clipper.clip)
            torch.nn.utils.clip_grad_norm(text_net.parameters(), text_clipper.clip)
        
        optimizer.step()
        # add loss to average
        train_loss += loss.data
        print(train_loss.cpu()[0]/num_batches)
    return train_loss/num_batches

def test_epoch(img_net, text_net, f_nodes, batch_size):
    # set to evaluation mode
    img_net.eval()
    text_net.eval()
    # for keeping track of the average loss
    test_batches = 0
    test_loss = 0
    for batch in batcher(f_nodes, batch_size, args.visual, args.audio, shuffle = False):
        img, text = batch 
        test_batches += 1
        # convert data to pytorch variables
        img, text = Variable(dtype(img)), Variable(dtype(text))
        # embed the images and audio using the networks
        img_embedding = img_net(img)
        text_embedding = text_net(text)
        loss = batch_hinge_loss(img_embedding, text_embedding, cuda)
        # add loss to average
        test_loss += loss.data 
    return test_loss/test_batches 
epoch = 1

################################# training/test loop #####################################

# run the training loop for the indicated amount of epochs 
while epoch <= args.n_epochs:
    # keep track of runtime
    start_time = time.time()

    print('training epoch: ' + str(epoch))
    # Train on the train set
    train_loss = train_epoch(epoch, img_net, text_net, optimizer, train, args.batch_size)
    # evaluate on the validation set
    val_loss = test_epoch(img_net, text_net, val, args.batch_size)
    
    # save network parameters
    save_params(img_net, 'img_model', epoch)
    save_params(text_net, 'text_model', epoch)

    # calculate the recall@n
    # create a minibatcher over the validation set
    iterator = batcher(val, args.batch_size, args.visual, args.audio, shuffle = False)
    # calc recal, pass it the iterator, the embedding functions and n
    # returns the measures columnise (speech2image retrieval) and rowwise(image2speech retrieval)
    recall, median_rank = speech2image(iterator, img_net, text_net, [1, 5, 10], dtype)

    # print some info about this epoch
    print("Epoch {} of {} took {:.3f}s".format(
            epoch, args.n_epochs, time.time() - start_time))
    print("training loss:\t\t{:.6f}".format(train_loss.cpu()[0]))
    print("validation loss:\t\t{:.6f}".format(val_loss.cpu()[0]))
    
    print('recall@1 = ' + str(recall[0]*100) + '%')
    print('recall@5 = ' + str(recall[1]*100) + '%')
    print('recall@10 = ' + str(recall[2]*100) + '%')
    print('median rank= ' + str(median_rank))
    epoch += 1
    if args.gradient_clipping:
        text_clipper.update_clip_value()
        text_clipper.reset_gradients()
        img_clipper.update_clip_value()
        img_clipper.reset_gradients()
    
test_loss = test_epoch(img_net, text_net, test, args.batch_size)
# calculate the recall@n
# create a minibatcher over the test set
iterator = batcher(test, args.batch_size, args.visual, args.audio, shuffle = False,)
# calc recal, pass it the iterator, the embedding functions and n
# returns the measures columnise (speech2image retrieval) and rowwise(image2speech retrieval)
recall, median_rank = speech2image(iterator, img_net, text_net, [1, 5, 10], dtype)
print("test loss:\t\t{:.6f}".format(test_loss.cpu()[0]))
print('test recall@1 = ' + str(recall[0]*100) + '%')
print('test recall@5 = ' + str(recall[1]*100) + '%')
print('test recall@10 = ' + str(recall[2]*100) + '%')
print('test median rank= ' + str(median_rank))

if args.gradient_clipping:
    text_clipper.save_grads(args.results_loc, 'textgrads')
    img_clipper.save_grads(args.results_loc, 'imgrads')
