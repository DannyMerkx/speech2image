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

from minibatchers import iterate_minibatches
from costum_loss import cosine_hinge_loss, dot_hinge_loss, l2norm_hinge_loss
from evaluate import calc_recall_at_n
## implementation of an CNN for af recognition. made for use with mel filterbank features
parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')
#parser.add_argument('-load_weights', type = bool, default = False, 
#                    help = 'load a pre-trained model (True), or initialise a new one (False), default: False')
#parser.add_argument('-weight_loc', type = str, default = './model.npz',
#                    help = 'location of pretrained weights, default: ./model.npz ')
parser.add_argument('-data_loc', type = str, default = '/train_data/flickr_features.h5',
                    help = 'location of the feature file, default: /data/processed/fbanks.h5')
parser.add_argument('-batch_size', type = int, default = 128, help = 'batch size, default: 128')
parser.add_argument('-lr', type = float, default = 0.00005, help = 'learning rate, default:0.00005')
parser.add_argument('-data_base', type = str, default = 'places', help = 'database to train on')
#parser.add_argument('-test', type = bool, default = False, 
#                    help = 'set to True to skip training and only run the network on the test set, use in combination with a pretrained model of course, default: False')
#parser.add_argument('-feat_type', type = str, default = 'fbanks', 
#                    help = 'type of input feature, either mfcc, fbanks, freq_spectrum or raw, default: fbanks')

args = parser.parse_args()

# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 
#get a list of all the nodes in the file
def iterate_places(h5_file):
    for x in h5_file.root:
        for y in x:
            yield y
            
def iterate_flickr(h5_file):
    for x in h5_file.root:
        yield x

if args.data_base == 'places':
    f_nodes = [node for node in iterate_places(data_file)]
elif args.data_base == 'flickr':
    f_nodes = [node for node in iterate_flickr(data_file)]
else:
    print('incorrect database')
    exit()
# total number of nodes (i.e. files) 
n_nodes= len(f_nodes)

# shuffle before dividing into train test and validation sets
np.random.shuffle(f_nodes)

train = f_nodes[0:7000]
val = f_nodes[7000:7500]
test = f_nodes[7500:8000]

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
    
    # loss function
    loss = l2norm_hinge_loss(img_embedding, speech_embedding)
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
    
    # validation function 
    test_loss = l2norm_hinge_loss(test_embed_img, test_embed_speech)
    test_loss = test_loss.mean()
   
    test_fn = theano.function([input_var_img, input_var_audio], test_loss)
    
    #############################################################

    # Finally, launch the training loop.
    print("Starting training...")
    epoch=1
    num_epochs = 50
    while epoch < num_epochs:
    ################################
    # learning rate scheme
        lr_shared.set_value(args.lr * (0.5 ** (epoch // 5)))

    ################################################################
       # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        print('epoch: '+ str(epoch+1))
        
        for batch in iterate_minibatches(train, args.batch_size, shuffle = True):
            img, speech = batch
            train_err += train_fn(img, speech)
            train_batches += 1
        print('train error: ' + str(train_err/train_batches))
        # optionally save the network parameters after each epoch
        #np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
        
        # full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(val,args.batch_size, shuffle = False):
            img, speech = batch
            val_err += test_fn(img, speech)
            val_batches += 1
        print('validation error: ' + str(val_err/val_batches))
        # calculate the recall@n
        # create an minibatcher over the validation set
        iterator = iterate_minibatches(val,args.batch_size, shuffle = False)
        # calc recal, pass it the iterator, the embedding functions and n
        recall, avg_rank = calc_recall_at_n(iterator, speech_out_fn, img_out_fn, [1, 5, 10])

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("validation loss:\t\t{:.6f}".format(val_err / val_batches))
        #print("  validation accuracy:\t\t{:.2f} %".format(
          #  val_acc / val_batches * 100))
        epoch=epoch+1
       
        print('recall@1 = ' + str(recall[0]*100) + '%')
        print('recall@5 = ' + str(recall[1]*100) + '%')
        print('recall@10 = ' + str(recall[2]*100) + '%')
        print('average rank= ' + str(avg_rank))
    # After training, we compute and print the test error:
    print ('computing test accuracy')
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(test, args.batch_size, shuffle = False):
        img, speech = batch
        test_err += test_fn(img, speech)
        test_batches += 1
    print('test error: ' + str(test_err/test_batches))


    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    iterator = iterate_minibatches(val,args.batch_size, shuffle = False)
    recall, avg_rank = calc_recall_at_n(iterator, speech_out_fn, img_out_fn, [1, 5, 10])
    print('Test recall@1 = ' + str(recall[0]*100) + '%')
    print('Test recall@5 = ' + str(recall[1]*100) + '%')
    print('Test recall@10 = ' + str(recall[2]*100) + '%')
    print('Test average rank= ' + str(avg_rank))
#    print("Final results:")
#    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
#    print("  test accuracy:\t\t{:.2f} %".format(
#        test_acc / test_batches * 100))
#    # save predictions targets and model weights
#    np.savez('predictions_' + args.af + '.npz', preds)
#    np.savez('targets_' + args.af + '.npz', targs)
#    np.savez(args.af + '_model.npz', *lasagne.layers.get_all_param_values(network))

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    main()
