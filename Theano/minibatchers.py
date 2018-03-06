#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:06:06 2018

@author: danny
minibatchers for the neural network training. to-do: features to take are currently
hard coded, implement something so that the features to extract from the h5 file
can be passed as arguments
"""
import numpy as np

# minibatch iterator which pads and truncates the inputs to a given size.
# slighly faster to pad on the go than to load a bigger dataset, but impractical
# when you want normalisation to also apply to the padding for instance
def iterate_minibatches_resize(f_nodes, batchsize, input_size, shuffle=True):  
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):  
        # take a batch of nodes of the given size               
        excerpt = f_nodes[start_idx:start_idx + batchsize]        
        speech=[]
        images=[]
        for ex in excerpt:
            # extract and append the vgg16 features
            images.append(ex.vgg._f_list_nodes()[0][:])
            # randomly choose a speech file. transpose the features if # frames is not yet the last dimension
            sp = np.random.choice(ex.fbanks._f_list_nodes())[:].transpose()
            # padd to the given input size
            while np.shape(sp)[1] < input_size:
                sp = np.concatenate((sp, np.zeros([40,1])),1)
            # truncate to the given input size
            if np.shape(sp)[1] >input_size:
                sp = sp[:,:input_size]
            speech.append(sp)
        # reshape the features into appropriate shape and recast as float32
        speech_shape = np.shape(speech)
        # for the speech features the network expects inputs of dim(batch_size, #channels, #fbanks, #frames(1024))
        speech = np.float32(np.reshape(speech,(speech_shape[0],1,speech_shape[1],speech_shape[2])))
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float32(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, speech    

# minibatch iterator which assumes inputs of uniform size
def iterate_minibatches(f_nodes, batchsize, shuffle=True):  
    if shuffle:
        # optionally shuffle the input during training
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size               
        excerpt = f_nodes[start_idx:start_idx + batchsize]       
        speech=[]
        images=[]
        for ex in excerpt:
            # extract and append the vgg16 features
            images.append(ex.vgg._f_list_nodes()[0][:])
            # get the number of speech files for this image to do random sampling
            n = np.shape(ex.fbanks._f_list_nodes())[0]
            # extract and append randomly one of the speech file features (transpose if the #frames is not yet the last dimension)
            speech.append(ex.fbanks._f_list_nodes()[np.random.randint(0,n)][:].transpose())
        # reshape the features into appropriate shape and recast as float32
        speech_shape = np.shape(speech)
        # for the speech features the network expects inputs of dim(batch_size, #channels, #fbanks, #frames(1024))
        speech = np.float32(np.reshape(speech,(speech_shape[0],1,speech_shape[1],speech_shape[2])))
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float32(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, speech   