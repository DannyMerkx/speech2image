#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:06:06 2018

@author: danny
"""
import numpy as np

# minibatch iterator which pads and truncates the inputs to a given size
def iterate_minibatches_resize(f_nodes, batchsize, input_size, shuffle=True):  
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):               
        excerpt = f_nodes[start_idx:start_idx + batchsize]        
        speech=[]
        images=[]
        for ex in excerpt:
            images.append(ex.vgg._f_list_nodes()[0][:])
            sp = np.random.choice(ex.fbanks._f_list_nodes())[:].transpose()
            while np.shape(sp)[1] < input_size:
                sp = np.concatenate((sp, np.zeros([40,1])),1)
            if np.shape(sp)[1] >input_size:
                sp = sp[:,:input_size]
            speech.append(sp)
            #audio.append(np.random.choice(ex.fbanks._f_list_nodes())[:])
        speech_shape = np.shape(speech)
        speech = np.float32(np.reshape(speech,(speech_shape[0],1,speech_shape[1],speech_shape[2])))
        images_shape = np.shape(images)
        images = np.float32(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, speech    
# minibatch iterator which assumes inputs of uniform size
def iterate_minibatches(f_nodes, batchsize, shuffle=True):  
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):               
        excerpt = f_nodes[start_idx:start_idx + batchsize]        
        speech=[]
        images=[]
        for ex in excerpt:
            images.append(ex.vgg._f_list_nodes()[0][:])
            # get the number of speech files for this image to do random sampling
            n = np.shape(ex.fbanks._f_list_nodes())[0]
            speech.append(ex.fbanks._f_list_nodes()[np.random.randint(0,n)][:].transpose())
        speech_shape = np.shape(speech)
        speech = np.float32(np.reshape(speech,(speech_shape[0],1,speech_shape[1],speech_shape[2])))
        images_shape = np.shape(images)
        images = np.float32(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, speech   