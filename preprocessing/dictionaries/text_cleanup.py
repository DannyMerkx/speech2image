#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 12:36:40 2018
Some functions for tokenising captions and filtering tokens from captions
@author: danny
"""
from nltk.tokenize.nist import NISTTokenizer
from contractions import contractions
import string

# replace contractions in the captions with the uncontracted forms in a contraction dictionary
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

# given a spelling dictionary of spelling corrections for the coprus' misspells,
# replace the misspelled words.
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
    
# remove all words that have a low occurrence in the dataset replace with oov
def remove_low_occurence(caption, dictionary, threshold, token):
    cleaned_capt = []
    keys = dictionary.keys()
    for x in caption:
        if not x in keys:
            x = token
        elif dictionary[x] < threshold:
            x = token
        cleaned_capt.append(x)
    return cleaned_capt
# remove all stop words and replace with stop word token
def remove_stop_words(caption, stop_words, token):
    cleaned_capt = []
    for x in caption:
        if x in stop_words:
            x = token
        cleaned_capt.append(x)
    return cleaned_capt
# remove all tokens with numerical values in them and replaces with a num token
def remove_numerical(caption, token):
    cleaned_capt = []
    for x in caption:
        for y in string.digits:
            if y in x:
                x = token
        cleaned_capt.append(x)
    return cleaned_capt   
# replace punctuation with a punctuation token
def remove_punctuation(caption, token):
    cleaned_capt = []
    for x in caption:
        if x in string.punctuation:
            x = token
        cleaned_capt.append(x)
    return cleaned_capt