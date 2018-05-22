#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:56:15 2018

@author: danny

Functions to prepare text data, e.g. one hot encoding and embedding indices.
"""
import string
import numpy as np
import pickle 

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

def find_index(char):
    # define the set of valid characters.
    valid_chars = string.printable
    return valid_chars.find(char)

def char_2_1hot(raw_text, batch_size, max_sent_len):
    n_letters = len(string.printable)
    text_batch = np.zeros([batch_size, max_sent_len, n_letters])
    # keep track of the origin sentence length to use in pack_padded_sequence
    lengths = []
    for i, text in enumerate(raw_text):
        lengths.append(len(text))        
        for j, char in enumerate(text):
            text_batch[i][j][find_index(char)] = 1
    return text_batch, lengths


def char_2_index(raw_text, batch_size, max_sent_len):
    text_batch = np.zeros([batch_size, max_sent_len])
    # keep track of the origin sentence length to use in pack_padded_sequence
    lengths = []
    for i, text in enumerate(raw_text):
        lengths.append(len(text))        
        for j, char in enumerate(text):
            text_batch[i][j] = find_index(char)
    return text_batch, lengths

def word_2_index(word_list, batch_size, max_sent_len, dict_loc):
    w_dict = load_obj(dict_loc)
    text_batch = np.zeros([batch_size, max_sent_len])
    lengths = []
    for i, words in enumerate(word_list):
        lengths.append(len(words))
        for j, word in enumerate(words):
            text_batch[i][j] = w_dict[word]
    return text_batch, lengths