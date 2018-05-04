#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:56:15 2018

@author: danny
"""
import string
import numpy as np

def find_index(char, valid_chars):  
    return valid_chars.find(char)

def char_2_1hot(raw_text, batch_size, valid_chars = string.printable):
    n_letters = len(valid_chars)
    text_batch = np.zeros([batch_size, len(raw_text), n_letters])
    for i, text in enumerate(raw_text):        
        for j, char in enumerate(text):
            text_batch[i][j][find_index(char, valid_chars)] = 1
    return text_batch


def char_2_index(raw_text, batch_size, max_sent_len, valid_chars = string.printable):
    text_batch = np.zeros([batch_size, max_sent_len])
    for i, text in enumerate(raw_text):        
        for j, char in enumerate(text):
            text_batch[i][j] = find_index(char, valid_chars)
    return text_batch