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
    # define the set of valid characters. use only lower case + 0, 0 will 
    # be a special character used as padding for which the embedding will remain
    # set to 0
<<<<<<< HEAD
    #valid_chars = '0^!?,.:' + string.ascii_lowercase
    #index = valid_chars.find(char)
    #if index < 0:
    #    # map all whitespace and punctuation to a single index
    #    if string.digits.find(char) >= 0:
    #        return len(valid_chars)
    #    if string.punctuation.find(char) >= 0:
    #        return len(valid_chars) + 1
    #    if string.whitespace.find(char) >= 0:
    #        return len(valid_chars) + 2
    valid_chars = string.printable
=======
    valid_chars = '0^!?,.:' + string.ascii_lowercase
    index = valid_chars.find(char)
    if index < 0:
        # map all whitespace and punctuation to a single index
        if string.digits.find(char) >= 0:
            return len(valid_chars)
        if string.punctuation.find(char) >= 0:
            return len(valid_chars) + 1
        if string.whitespace.find(char) >= 0:
            return len(valid_chars) + 2
    
>>>>>>> 9ed6340bbfd703bf0a13889b7f6a5b9f30b05aca
    return valid_chars.find(char)

def char_2_1hot(raw_text, batch_size, valid_chars = string.printable):
    n_letters = len(valid_chars)
    text_batch = np.zeros([batch_size, len(raw_text), n_letters])
    for i, text in enumerate(raw_text):        
        for j, char in enumerate(text):
            text_batch[i][j][find_index(char, valid_chars)] = 1
    return text_batch


def char_2_index(raw_text, batch_size, max_sent_len):
    text_batch = np.zeros([batch_size, max_sent_len])
    for i, text in enumerate(raw_text):        
        for j, char in enumerate(text):
            text_batch[i][j] = find_index(char)
<<<<<<< HEAD
    return text_batch
=======
    return text_batch
>>>>>>> 9ed6340bbfd703bf0a13889b7f6a5b9f30b05aca
