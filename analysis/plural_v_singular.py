#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:27:54 2021

@author: danny
"""
import pandas as pd

noun_annot = pd.read_csv(open('../Image_annotations/annotations/Noun_annotations.csv'), 
                          index_col=0)
verb_annot = pd.read_csv(open('../Image_annotations/annotations/Verb_annotations.csv'), 
                          index_col=0)

data_loc = './lmer.csv'

results = pd.read_csv(data_loc, index_col=0)

results = results[results['full_word']]

results = results[(results['s_id'] == 's1_1') | (results['s_id'] == 's2_1')]

results = results[(results['word_form'] == 'singular') | (results['word_form'] == 'plural')]

results = results[results['word'] != 'water']
results = results[results['word'] != 'grass']
results = results[results['word'] != 'snow']
results = results[results['word'] != 'soccer']
results = results[results['word'] != 'shorts']
results = results[results['word'] != 'sunglasses']

results[(results['word_form'] == 'plural') & (results['VQ']==0)]['p@10'].mean()
results[(results['word_form'] == 'plural') & (results['VQ']==1)]['p@10'].mean()

results[(results['word_form'] == 'singular') & (results['VQ']==0)]['p@10'].mean()
results[(results['word_form'] == 'singular') & (results['VQ']==1)]['p@10'].mean()

