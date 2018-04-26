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
from prep_text import text_2_1hot, text_2_index
# minibatcher which takes a list of nodes and returns the visual and audio features, possibly resized.
# visual and audio should contain a string of the names of the visual and audio features nodes in the h5 file.
# input size is the number of input features, frames is the length of the sequence
def iterate_minibatches(f_nodes, batchsize, visual, audio, frames = 1024, shuffle=True):  
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
            images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
            # get the number of speech files for this image to do random sampling
            n = np.shape(ex.fbanks._f_list_nodes())[0]
            # randomly choose a speech file. transpose the features if # frames is not yet the last dimension
            sp = eval('ex.' + audio + '._f_list_nodes()[np.random.randint(0,n)].read().transpose()')
            # padd to the given output size
            n_frames = sp.shape[1]
            if n_frames < frames:
                sp = np.pad(sp, [(0, 0), (0, frames - n_frames )], 'constant')
            # truncate to the given input size
            if n_frames > frames:
                sp = sp[:,:frames]
            speech.append(sp)
        # reshape the features into appropriate shape and recast as float32
        speech = np.float64(speech)
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, speech    

# the flickr minibatcher returns all 5 captions per image.
def iterate_minibatches_flickr(f_nodes, batchsize, visual, audio, frames = 1024, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for i in range(0,5):
        for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
            # take a batch of nodes of the given size               
            excerpt = f_nodes[start_idx:start_idx + batchsize]
            speech=[]
            images=[]
            for ex in excerpt:
                # extract and append the vgg16 features
                images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
                # extract the audio features
                sp = eval('ex.' + audio + '._f_list_nodes()[i].read().transpose()')
                # extract the audio features
                sp = eval('ex.' + audio + '._f_list_nodes()[i][:].transpose()')
                # padd to the given output size
                n_frames = sp.shape[1]
                if n_frames < frames:
                    sp = np.pad(sp, [(0, 0), (0, frames - n_frames )], 'constant')
                # truncate to the given input size
                if n_frames > frames:
                    sp = sp[:,:frames]
                speech.append(sp)
            # reshape the features and recast as float64
            speech = np.float64(speech)
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, speech

def iter_text_flickr(f_nodes, batchsize, visual, text, shuffle=True, test = False):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    if not test:
        for i in range(0,5):
            for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
                # take a batch of nodes of the given size               
                excerpt = f_nodes[start_idx:start_idx + batchsize]
                caption = []
                images = []
                for ex in excerpt:
                    # extract and append the vgg16 features
                    images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
                    # extract the audio features
                    cap = eval('ex.' + text + '._f_list_nodes()[i].read()')
                    cap = cap.decode('utf-8')
                    caption.append(cap)
                caption = text_2_index(caption, batchsize, 200)
                images_shape = np.shape(images)
                # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
                images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
                yield images, caption
    if test:
        for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
                # take a batch of nodes of the given size               
                excerpt = f_nodes[start_idx:start_idx + batchsize]
                caption = []
                images = []
                for ex in excerpt:
                    # extract and append the vgg16 features
                    images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
                    # extract the audio features
                    cap = eval('ex.' + text + '._f_list_nodes()[0].read()')
                    cap = cap.decode('utf-8')
                    caption.append(cap)
                caption = text_2_index(caption, batchsize, 200)
                images_shape = np.shape(images)
                # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
                images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
                yield images, caption