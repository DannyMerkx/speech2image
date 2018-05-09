#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:56:15 2018

@author: danny

Functions to prepare text data, e.g. one hot encoding and embedding indices.
"""
import string
import numpy as np

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
