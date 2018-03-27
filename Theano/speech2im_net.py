#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:20:04 2018

@author: danny
script for training neural networks which embed speech and images into the same
vector space. To-do: create balanced train, test and validation sets. Move some hard 
coding to the argparser
"""

#!/usr/bin/env python
from __future__ import print_function

import sys
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import tables
import argparse

from minibatchers import iterate_minibatches, iterate_minibatches_resize
from costum_loss import batch_hinge_loss
from evaluate import speech2image, image2speech

## implementation of a DNN structure for speech2im embedding/retrieval in Theano+Lasagne
parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')
parser.add_argument('-data_loc', type = str, default = '/prep_data/flickr_features.h5',
                    help = 'location of the feature file, default: /data/processed/fbanks.h5')
parser.add_argument('-batch_size', type = int, default = 128, help = 'batch size, default: 128')
parser.add_argument('-lr', type = float, default = 0.00005, help = 'learning rate, default:0.00005')
parser.add_argument('-data_base', type = str, default = 'places', help = 'database to train on, options: places, flickr')
parser.add_argument('-loss', type = list, default = [True, False], help = 'determines which embeddings are normalised by the loss function')
parser.add_argument('-n_epochs', type = int, default = 50, help = 'number of epochs, default: 50')
parser.add_argument('-data_split', type = list, default = [.9, .05, .05], help = 'split of the dataset into train, val and test respectively. Make sure it adds up to 1')

args = parser.parse_args()

# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 

#get a list of all the nodes in the file. The places database was so big 
# it needs to be split into subgroups at the root node so the iterator needs
# to look one node deeper in the tree.
def iterate_places(h5_file):
    for x in h5_file.root:
        for y in x:
            yield y
def iterate_flickr(h5_file):
    for x in h5_file.root:
        yield x
        
# define the batcher type to use.
batcher = iterate_minibatches_resize

if args.data_base == 'places':
    f_nodes = [node for node in iterate_places(data_file)]
elif args.data_base == 'flickr':
    f_nodes = [node for node in iterate_flickr(data_file)]
else:
    print('incorrect database option')
    exit()
    
# total number of nodes (i.e. speech/image files) 
n_nodes= len(f_nodes)

# shuffle before dividing into train test and validation sets
np.random.shuffle(f_nodes)

args.data_split = [int(np.floor(x * n_nodes)) for x in args.data_split]

train = f_nodes[0 : args.data_split[0]]
val = f_nodes[args.data_split[0] : args.data_split[1] + args.data_split[0]]
test = f_nodes[args.data_split[1] + args.data_split[0]: args.data_split[2] + args.data_split[1] + args.data_split[0]]

# the network for embedding the vgg16 features
def build_img_net(input_var=None):
    # input layer
    network = lasagne.layers.InputLayer(shape = (None, 4096),
                                        input_var = input_var)
    # output layer
    network = lasagne.layers.DenseLayer(network,
            num_units = 1024, W = lasagne.init.GlorotUniform(),
            nonlinearity = None)

    return network

# network for embedding the spoken captions
def build_audio_net(input_var=None):
    network = lasagne.layers.InputLayer(shape = (None, 1, 40, 1024),
                                        input_var = input_var)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, 
                                         filter_size=(40, 5), stride=(1,1), 
                                         pad='valid', W = lasagne.init.GlorotUniform(),
                                         nonlinearity= lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.FlattenLayer(network,outdim=3)
    
    network = lasagne.layers.MaxPool1DLayer(network,pool_size = (4), stride = (2), ignore_border = True)  
    
    network = lasagne.layers.Conv1DLayer(network, num_filters=512, filter_size=25,
                                         stride=1, pad='same', W=lasagne.init.GlorotUniform(), 
                                         nonlinearity=lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.MaxPool1DLayer(network,pool_size = (4), stride = (2), ignore_border = True)
    
    network = lasagne.layers.Conv1DLayer(network, num_filters=1024, filter_size=25,
                                         stride=1, pad='same', W=lasagne.init.GlorotUniform(), 
                                         nonlinearity=lasagne.nonlinearities.rectify)
    
    #network = lasagne.layers.FeaturePoolLayer(network, pool_size=253, axis = 2, pool_function=theano.tensor.max)
    network = lasagne.layers.MaxPool1DLayer(network,pool_size = lasagne.layers.get_output_shape(network, input_shapes=None)[-1], stride = 1, ignore_border = True)  
    network = lasagne.layers.ReshapeLayer(network,([0],[1]))
    
    return network    

# ############################## Main program ################################

def main(num_epochs = 50):
    # Prepare Theano variables for inputs and targets
    # image input tensor    
    input_var_img = T.matrix('img_inputs')
    # speech input tensor
    input_var_audio = T.tensor4('speech_inputs')
    
    # build networks    
    img_network = build_img_net(input_var_img)    
    speech_network = build_audio_net(input_var_audio)
    
    ###############TRAIN##################
    
    # functions to get img and speech embeddings
    img_embedding = lasagne.layers.get_output(img_network)
    speech_embedding = lasagne.layers.get_output(speech_network)
    
    # loss function. the args.loss argument determines which embeddings are normalised and thereby the distance measure. 
    # setting all to true results in the cosine similarity, all false results in the dot product. Harwath and glass
    # found it was best to only normalise the speech embeddings
    loss = batch_hinge_loss(img_embedding, speech_embedding, args.loss)
    loss = loss.mean()
    
    # create the parameters and update functions
    params = lasagne.layers.get_all_params([img_network,speech_network], trainable=True)
    # learning rate as a shared variable so that it can be adapted over time
    lr_shared = theano.shared(np.array(args.lr, dtype = theano.config.floatX))
    updates = lasagne.updates.momentum(
            loss, params, learning_rate=lr_shared, momentum=0.9)

    #training function
    train_fn = theano.function([input_var_img, input_var_audio], loss, updates=updates,allow_input_downcast=True)
    #########################################################
    
    ############## TEST ######################################
    # test functions
    # test_embed functions (set to deterministic to turn off features like drop-out)
    # can be used for the test loss or to embed unseen data (e.g. for calculating the recall).
    test_embed_img = lasagne.layers.get_output(img_network, deterministic=True)
    img_out_fn = theano.function([input_var_img], test_embed_img)
    
    test_embed_speech = lasagne.layers.get_output(speech_network, deterministic=True)
    speech_out_fn = theano.function([input_var_audio], speech_embedding)
    
    test_loss = batch_hinge_loss(test_embed_img, test_embed_speech, args.loss)
    test_loss = test_loss.mean()
   
    test_fn = theano.function([input_var_img, input_var_audio], test_loss)
    
    #############################################################

    # Finally, launch the training loop.
    print("Starting training...")
    epoch=0
    while epoch < args.n_epochs:
        # learning rate scheme
        lr_shared.set_value(args.lr * (0.5 ** (epoch // 5)))

    ################################################################
       # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        print('epoch: '+ str(epoch+1))
        
        for batch in batcher(train, args.batch_size, shuffle = True):
            img, speech = batch
            train_err += train_fn(img, speech)
            train_batches += 1
        # optionally save the network parameters after each epoch
        #np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
        
        # full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in batcher(val,args.batch_size, shuffle = False):
            img, speech = batch
            val_err += test_fn(img, speech)
            val_batches += 1
        # calculate the recall@n
        # create a minibatcher over the validation set
        iterator = batcher(val, args.batch_size, shuffle = False)
        # calc recal, pass it the iterator, the embedding functions and n
        recall, avg_rank = speech2image(iterator, img_out_fn, speech_out_fn, [1, 5, 10])

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        epoch = epoch + 1  
        print("training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print('recall@1 = ' + str(recall[0]*100) + '%')
        print('recall@5 = ' + str(recall[1]*100) + '%')
        print('recall@10 = ' + str(recall[2]*100) + '%')
        print('average rank= ' + str(avg_rank))
    # After training, we compute and print the test error:
    print ('computing test accuracy')
    test_err = 0
    test_batches = 0
    for batch in batcher(test, args.batch_size, shuffle = False):
        img, speech = batch
        test_err += test_fn(img, speech)
        test_batches += 1
    # calculate the recall
    iterator = batcher(val, args.batch_size, shuffle = False)
    recall, avg_rank = speech2image(iterator, img_out_fn, speech_out_fn, [1, 5, 10])
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print('Test recall@1 = ' + str(recall[0]*100) + '%')
    print('Test recall@5 = ' + str(recall[1]*100) + '%')
    print('Test recall@10 = ' + str(recall[2]*100) + '%')
    print('Test average rank= ' + str(avg_rank))
    
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural networks for joint image/caption embedding in Lasagne.")
    main()
