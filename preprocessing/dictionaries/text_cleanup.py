#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 12:36:40 2018

@author: danny
"""
from nltk.tokenize.nist import NISTTokenizer
from contractions import contractions
from nltk.corpus import wordnet

def rep_contractions(cap, contract):   
    for x in contract.keys():
        cap = cap.replace(x, contract[x])
    return(cap)

# apply the nist tokeniser and lower case if needed
def tokenise(caption, lower = True):
    # import the NIST tokenizer    
    nist = NISTTokenizer() 
    if lower:
        caption = caption.lower()    
    caption = nist.tokenize(caption)
    return caption

def correct_spel(caption, spell_dict):
    contract = contractions()
    cleaned_capt = [] 
    for x in caption:
        x = rep_contractions(x, contract)
        x = x.replace(' ave.', ' avenue').replace(' ave ', ' avenue ').replace('&', 'and').replace('=', '')
        
        if x in spell_dict:
            x = spell_dict[x]
        cleaned_capt.append(x)
    return(cleaned_capt)
    
# remove all words that occur only once in the dataset replace with oov
def remove_low_occurence(caption, dictionary):
    cleaned_capt = []
    keys = dictionary.keys()
    for x in caption:
        if not x in keys:
            x = '<oov>'
        elif dictionary[x] < 5:
            x = '<oov>'
        cleaned_capt.append(x)
    return cleaned_capt
    
# remove all stop words and replace with stop word token
def remove_stop_words(caption, stop_words):
    cleaned_capt = []
    for x in caption:
        if x in stop_words:
            x = '<stop>'
        cleaned_capt.append(x)
    return cleaned_capt

def remove_numerical(caption, digits):
    cleaned_capt = []
    for x in caption:
        for y in digits:
            if y in x:
                x = '<num>'
        cleaned_capt.append(x)
    return cleaned_capt   
    
# replace punctuation with a punctuation token
def remove_punctuation(caption, punct):
    cleaned_capt = []
    for x in caption:
        if x in punct:
            x = '<punct>'
        cleaned_capt.append(x)
    return cleaned_capt
# remove all words that are not in wordnet and replace with oov
def clean_wordnet(caption):
    cleaned_capt = []
    for x in caption:
        if wordnet.synsets(x) == [] and not x == '<punct>' and not x == '<stop>' and not x == '<num>':
            x = '<oov>'
        cleaned_capt.append(x)
    return cleaned_capt
