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
import string
import numpy as np
import pickle
########################################################################################################
# the following functions are used to convert the input strings to indices for the word embedding layers

# loader for the dictionary, loads a pickled dictionary.
def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

# function to find the indices of characters. optionally pass a list of valid
# characters, otherwise just use string.printable
def find_index(char, valid_chars = False):
    # define the set of valid characters.
    if valid_chars == False:
        valid_chars = string.printable 
    idx = valid_chars.find(char)
    # add 1 to the index so the range is 1 to 101 and 0 is free to use as padding idx
    if idx != -1:
        idx += 1
    return idx

# function to turn the character based text caption into character indices
def char_2_index(batch, batch_size):
    # get the maximum sentence length, the entire batch must be padded to this length
    max_sent_len = max([len(x) for x in batch])
    index_batch = np.zeros([batch_size, max_sent_len])
    # keep track of the original sentence length to use in pack_padded_sequence
    lengths = []
    for i, text in enumerate(batch):
        lengths.append(len(text))        
        for j, char in enumerate(text):
            index_batch[i][j] = find_index(char)
    return index_batch, lengths

def word_2_index(batch, batch_size, dict_loc):
    w_dict = load_obj(dict_loc)
    # filter words that do not occur in the dictionary
    batch = [[word if word in w_dict else '<oov>' for word in sent] for sent in batch]
    # get the maximum sentence length, the entire batch must be padded to this length
    max_sent_len = max([len(x) for x in batch])
    index_batch = np.zeros([batch_size, max_sent_len])
    lengths = []
    # load the indices for the words from the dictionary
    for i, words in enumerate(batch):
        lengths.append(len(words))
        for j, word in enumerate(words):
            index_batch[i][j] = w_dict[word]
    return index_batch, lengths
##############################################################################################################
################################### minibatchers ############################################################

# minibatcher which takes a list of nodes and returns the visual and audio features, possibly resized.
# visual and audio should contain a string of the names of the visual and audio features nodes in the h5 file.
# frames is the max length of the time sequence, the batcher truncates to this length.
def iterate_audio(f_nodes, batchsize, visual, audio, shuffle=True):  
    frames = 2048
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
            # extract and append the visual features
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
      
        max_length = max(lengths)
        # reshape the features and recast as float64
        speech = np.float64(speech)
        # truncate all padding to the length of the longest utterance
        speech = speech[:,:, :max_length]
        # reshape the features into appropriate shape and recast as float32
        speech = np.float64(speech)
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, speech, lengths  

# batcher for character input. Keeps track of the unpadded senctence lengths to use with 
# pytorch's pack_padded_sequence. Optionally shuffle.
def iterate_char(f_nodes, batchsize, visual, text, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size               
        excerpt = f_nodes[start_idx:start_idx + batchsize]
        caption = []
        images = []
        for ex in excerpt:
            # extract and append the visual features
            images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
            # extract the audio features
            cap = eval('ex.' + text + '._f_list_nodes()[i].read()')
            cap = cap.decode('utf-8')
            # append an otherwise unused character as a start of sentence character and 
            # convert the sentence to lower case.
            caption.append(cap)
        # converts the sentence to character ids. 
        caption, lengths = char_2_index(caption, batchsize)
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, caption, lengths

# batcher for token input. Keeps track of the unpadded senctence lengths to use with 
# pytorch's pack_padded_sequence. Requires a pre-defined dictionary mapping the tokens
# to indices. Optionally shuffle.
def iterate_tokens(f_nodes, batchsize, visual, text, dict_loc, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size               
        excerpt = f_nodes[start_idx:start_idx + batchsize]
        caption = []
        images = []
        for ex in excerpt:
            # extract and append the visual features
            images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
            # extract the audio features
            cap = eval('ex.' + text + '._f_list_nodes()[i].read()')
            cap = [x.decode('utf-8') for x in cap]
            # append an otherwise unused character as a start of sentence character and 
            # convert the sentence to lower case.
            caption.append(cap)
        # converts the sentence to character ids. 
        caption, lengths = word_2_index(caption, batchsize, dict_loc)
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, caption, lengths


# batcher for audio input. Keeps track of the unpadded senctence lengths to use with 
# pytorch's pack_padded_sequence. Optionally shuffle.
def iterate_audio_5fold(f_nodes, batchsize, visual, audio, shuffle = True):
    max_frames = 2048
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
                # extract and append the visual features
                images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
                # extract the audio features
                sp = eval('ex.' + audio + '._f_list_nodes()[i].read().transpose()')
                # padd to the given output size
                n_frames = sp.shape[1]
		
                if n_frames < max_frames:
                    sp = np.pad(sp, [(0, 0), (0, max_frames - n_frames )], 'constant')
                # truncate to the given input size
                if n_frames > max_frames:
                    sp = sp[:,:max_frames]
                    n_frames = max_frames
                lengths.append(n_frames)
                speech.append(sp)
            max_length = max(lengths)
            # reshape the features and recast as float64
            speech = np.float64(speech)
            # truncate all padding to the length of the longest utterance
            speech = speech[:,:, :max_length]
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, speech, lengths

# batcher for character input. Keeps track of the unpadded senctence lengths to use with 
# pytorch's pack_padded_sequence. Optionally shuffle.
def iterate_char_5fold(f_nodes, batchsize, visual, text, shuffle=True):
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
                # extract and append the visual features
                images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
                # extract the audio features
                cap = eval('ex.' + text + '._f_list_nodes()[i].read()')
                cap = cap.decode('utf-8')
                # append an otherwise unused character as a start of sentence character and 
                # convert the sentence to lower case.
                caption.append(cap)
            # converts the sentence to character ids. 
            caption, lengths = char_2_index(caption, batchsize)
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, caption, lengths

# batcher for token input. Keeps track of the unpadded senctence lengths to use with 
# pytorch's pack_padded_sequence. Requires a pre-defined dictionary mapping the tokens
# to indices. Optionally shuffle.
def iterate_tokens_5fold(f_nodes, batchsize, visual, text, dict_loc, shuffle=True):
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
                # extract and append the visual features
                images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
                # extract the audio features
                cap = eval('ex.' + text + '._f_list_nodes()[i].read()')
                # add begin of sentence and end of sentence tokens
                cap = ['<s>'] + [x.decode('utf-8') for x in cap] + ['</s>']
                                
                caption.append(cap)
            # converts the sentence to character ids. 
            caption, lengths = word_2_index(caption, batchsize, dict_loc)
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, caption, lengths
