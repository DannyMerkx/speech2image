#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:49:51 2018

@author: danny
"""
import string
import sys
sys.path.append('/data/speech2image/preprocessing/dictionaries')

from text_cleanup import tokenise, correct_spel, remove_low_occurence, remove_stop_words, remove_punctuation, remove_numerical
from nltk.corpus import stopwords

def text_features_flickr(text_dict, output_file, append_name, node_list, freq_dict): 
    # threshold for the word occurrence frequency. Most previous papers remove words that do not occur more than 5 times
    # in the training data
    threshold = 5
    
    count = 1
    for node in node_list:
        print('processing file: ' + str(count))
        count+=1
        # create nodes for the raw text and the tokenised text
        raw_text_node = output_file.create_group(node, 'raw_text')
        token_node = output_file.create_group(node, 'tokens')
        clean_token_node = output_file.create_group(node, 'clean_tokens')
        # base name of the image caption pair to extract sentences from the dictionary
        base_name = node._v_name.split(append_name)[1]
        
        captions = text_dict[base_name]['sentences']
        for x in captions:            
            raw = x['raw']
            if not raw[-1] == '.':
                raw = raw +' .'
            tokens = x['tokens']
            clean_tokens = remove_low_occurence(tokens, freq_dict, threshold)
            output_file.create_array(raw_text_node, append_name + base_name + '_' + str(x['sentid']), bytes(raw, 'utf-8'))
            output_file.create_array(token_node, append_name +  base_name + '_' + str(x['sentid']), tokens) 
            output_file.create_array(clean_token_node, append_name +  base_name + '_' + str(x['sentid']), clean_tokens)
    
def text_features_coco(text_dict, output_file, append_name, node_list, spell_dict, freq_dict):     
    # make lists of stop words, digits and punctuation
    stop_words = stopwords.words('english')
    punct = string.punctuation
    digits = string.digits
    # threshold for the word occurrence frequency. Most previous papers remove words that do not occur more than 5 times
    # in the training data
    threshold = 5
    
    count = 1
    for node in node_list:
        print('processing file: ' + str(count))
        count+=1
        # create nodes for the raw text and the tokenised text
        raw_text_node = output_file.create_group(node, 'raw_text')
        token_node = output_file.create_group(node, 'clean_tokens')
        raw_token_node = output_file.create_group(node, 'raw_tokens')
        corrected_node = output_file.create_group(node, 'spell_tokens')
        cannonical_node = output_file.create_group(node, 'cannonical_tokens')
        # base name of the image caption pair to extract sentences from the dictionary
        base_name = node._v_name.split(append_name)[1]
        
        captions = text_dict[base_name]        
        for x in captions:
            # the raw caption is just the original text, tokenised to remove extra spaces etc. and place a dot at the 
            # end of every sentence.            
            raw = ''.join([' ' + y for y in tokenise(x['caption'], lower = False)])[1:]
            if not raw[-1] == '.':
                raw = raw +' .'
            # raw tokens are the raw caption with only tokenisation
            raw_tokens = tokenise(x['caption'])
            if not raw_tokens[-1] == '.':
                raw_tokens[-1] = '.'
            # tokens with simple spelling correction applied. 
            corrected_tokens = correct_spel(raw_tokens, spell_dict)
            # the most common tokens in other research use only words occuring over 5 times in the training data, remove numerical values and punctuation.
            cannonical_tokens = remove_punctuation(remove_low_occurence(remove_numerical(raw_tokens, digits), freq_dict, threshold), punct)
            # the cannonical tokens but also with stop words removed and uses the spell correct tokens
            clean_tokens = remove_punctuation(remove_stop_words(remove_low_occurence(remove_numerical(corrected_tokens, digits), freq_dict, threshold), stop_words), punct)
            
            output_file.create_array(raw_token_node, append_name +  base_name + '_' + str(x['id']), raw_tokens)
            output_file.create_array(corrected_node, append_name +  base_name + '_' + str(x['id']), corrected_tokens)
            output_file.create_array(raw_text_node, append_name + base_name + '_' + str(x['id']), bytes(raw, 'utf-8'))
            output_file.create_array(token_node, append_name +  base_name + '_' + str(x['id']), clean_tokens) 
            output_file.create_array(cannonical_node, append_name +  base_name + '_' + str(x['id']), cannonical_tokens)
