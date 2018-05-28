#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:06:06 2018

@author: danny
minibatchers for the neural network training. There are three batchers for speech, tokens and raw text.
Each batcher has a 5 fold version as many image captioning databases have multiple (5) captions per image.
The batchers also return the lenghts of the captions in the batch so it can be used with torch
pack_padded_sequence.
"""
import numpy as np
from prep_text import char_2_1hot, char_2_index
# minibatcher which takes a list of nodes and returns the visual and audio features, possibly resized.
# visual and audio should contain a string of the names of the visual and audio features nodes in the h5 file.
#frames is the desired length of the time sequence, the batcher pads or truncates.
def iterate_audio(f_nodes, batchsize, visual, audio, frames = 2048, shuffle=True):  
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):  
        # take a batch of nodes of the given size               
        excerpt = f_nodes[start_idx:start_idx + batchsize]        
        speech = []
        images = []
        lengths = []
        for ex in excerpt:
            # extract and append the vgg16 features
            images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
            # retrieve the audio features
            sp = eval('ex.' + audio + '._f_list_nodes()[0].read().transpose()')
            # padd to the given output size
            n_frames = sp.shape[1]
            if n_frames < frames:
                sp = np.pad(sp, [(0, 0), (0, frames - n_frames )], 'constant')
            # truncate to the given input size
            if n_frames > frames:
                sp = sp[:,:frames]
                n_frames = frames
            lengths.append(n_frames)
            speech.append(sp)
        # reshape the features into appropriate shape and recast as float32
        speech = np.float64(speech)
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, speech, lengths  

# visual and text should be the names of the feature nodes in the h5 file, chars is the maximum sentence length in characters.
# default is 260, to accomodate the max lenght found in mscoco. The max lenght in flickr is 200 
def iterate_raw_text(f_nodes, batchsize, visual, text, chars = 260, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
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
            # append an otherwise unused character as a start of sentence character and 
            # convert the sentence to lower case.
            caption.append(cap)
        # converts the sentence to character ids. 
        caption, lengths = char_2_index(caption, batchsize, chars)
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, caption, lengths

# slightly different from the raw text loader, also takes a dictionary location. Max words default is 60 to accomodate mscoco.
def iterate_tokens(f_nodes, batchsize, visual, text, dict_loc, words = 60, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
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
            cap = [x.decode('utf-8') for x in cap]
            # append an otherwise unused character as a start of sentence character and 
            # convert the sentence to lower case.
            caption.append(cap)
        # converts the sentence to character ids. 
        caption, lengths = word_2_index(caption, batchsize, words, dict_loc)
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, caption, lengths


# the 5fold minibatchers are for the datasets with 5 captions per image (mscoco, flickr). It returns all 5 captions per image.
def iterate_audio_5fold(f_nodes, batchsize, visual, audio, frames = 2048, shuffle = True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for i in range(0, 5):
        for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
            # take a batch of nodes of the given size               
            excerpt = f_nodes[start_idx:start_idx + batchsize]
            speech = []
            images = []
            lengths = []
            for ex in excerpt:
                # extract and append the vgg16 features
                images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
                # extract the audio features
                sp = eval('ex.' + audio + '._f_list_nodes()[i].read().transpose()')
                # padd to the given output size
                n_frames = sp.shape[1]
		
                if n_frames < frames:
                    sp = np.pad(sp, [(0, 0), (0, frames - n_frames )], 'constant')
                # truncate to the given input size
                if n_frames > frames:
                    sp = sp[:,:frames]
                    n_frames = frames
                lengths.append(n_frames)
                speech.append(sp)
            # reshape the features and recast as float64
            speech = np.float64(speech)
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, speech, lengths

# iterate over text input. the value for chars indicates the max sentence lenght in characters. Keeps track 
# of the unpadded senctence lengths to use with pytorch's pack_padded_sequence.
def iterate_raw_text_5fold(f_nodes, batchsize, visual, text, chars = 260, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
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
                # append an otherwise unused character as a start of sentence character and 
                # convert the sentence to lower case.
                caption.append(cap)
            # converts the sentence to character ids. 
            caption, lengths = char_2_index(caption, batchsize, chars)
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, caption, lengths

# iterate over text input. the value for chars indicates the max sentence lenght in characters. Keeps track 
# of the unpadded senctence lengths to use with pytorch's pack_padded_sequence. slightly different from the raw text loader
# as we need a word_2_index function and this also takes a dictionary
def iterate_tokens_5fold(f_nodes, batchsize, visual, text, dict_loc, words = 60, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
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
                cap = [x.decode('utf-8') for x in cap]
                # append an otherwise unused character as a start of sentence character and 
                # convert the sentence to lower case.
                caption.append(cap)
            # converts the sentence to character ids. 
            caption, lengths = word_2_index(caption, batchsize, words, dict_loc)
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, caption, lengths