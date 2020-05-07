#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:49:51 2018

@author: danny
"""
import string
import sys
sys.path.append('/data/speech2image/preprocessing/dictionaries')

from text_cleanup import tokenise

def text_features_flickr(text_dict, output_file, append_name, node_list): 
    count = 1
    for node in node_list:
        print(f'processing file: {count}')
        count+=1
        # create nodes for the raw text and the tokenised text
        raw_text_node = output_file.create_group(node, 'raw_text')
        token_node = output_file.create_group(node, 'tokens')
        # base name of the image caption pair to extract sentences from the
        # dictionary
        base_name = node._v_name.split(append_name)[1]
        
        captions = text_dict[base_name]['sentences']
        for x in captions:            
            raw = x['raw']
            if not raw[-1] == '.':
                raw = f'{raw} .'
            tokens = x['tokens']
            output_file.create_array(raw_text_node, f'{append_name}{base_name}_{x["sentid"]}', bytes(raw, 'utf-8'))
            output_file.create_array(token_node, f'{append_name}{base_name}_{x["sentid"]}', tokens) 
            
def text_features_coco(text_dict, output_file, append_name, node_list):     
    count = 1
    for node in node_list:
        print('processing file: ' + str(count))
        count+=1
        # create nodes for the raw text and the tokenised text
        raw_text_node = output_file.create_group(node, 'raw_text')
        token_node = output_file.create_group(node, 'tokens')
        # base name of the image caption pair to extract sentences from the dictionary
        base_name = node._v_name.split(append_name)[1]
        
        captions = text_dict[base_name]        
        for x in captions:
            # the raw caption is just the original text, tokenised to remove extra spaces etc. and place a dot at the 
            # end of every sentence.            
            raw = ''.join([' ' + y for y in tokenise(x['caption'], lower = False)])[1:]
            if not raw[-1] == '.':
                raw = raw +' .'
            # raw tokens are the captions with only tokenisation
            raw_tokens = tokenise(x['caption'])            
            capt = []
            for z in raw_tokens:
                z = ''.join([x for x in z if not x in string.punctuation])
                if not z == '':
                    capt.append(z)
            raw_tokens = capt
            output_file.create_array(token_node, append_name +  base_name + '_' + str(x['id']), raw_tokens)
            output_file.create_array(raw_text_node, append_name + base_name + '_' + str(x['id']), bytes(raw, 'utf-8'))
