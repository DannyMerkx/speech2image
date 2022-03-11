#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:33:48 2021
This was a little test that was added later to see how well plural nouns are recognised if you
drop the final phoneme i.e. the plural marker.
@author: danny
"""
import pandas as pd

data_loc = './lmer.csv'

results = pd.read_csv(data_loc, index_col=0)
results = results[(results['s_id'] == 's1_1') | (results['s_id'] == 's2_1')]


penultimate = results[(results['gate']) == results['n_gates']-1]

penultimate = penultimate[(penultimate['word_form'] == 'plural')]

results = results[results['full_word']]

results = results[results['word_form']== 'plural']

print(penultimate['p@10'].mean())
print(results['p@10'].mean())

print(penultimate[penultimate['VQ']==1]['p@10'].mean())
print(results[results['VQ']==1]['p@10'].mean())


print(penultimate[penultimate['VQ']!=1]['p@10'].mean())
print(results[results['VQ']!=1]['p@10'].mean())

results = pd.read_csv(data_loc, index_col=0)
results = results[(results['s_id'] == 's1_1') | (results['s_id'] == 's2_1')]

penultimate = results[(results['gate']) == results['n_gates']-1]

penultimate = penultimate[(penultimate['word_form'] == 'singular')]

results = results[results['full_word']]

results = results[results['word_form']== 'singular']

print(penultimate['p@10'].mean())
print(results['p@10'].mean())

print(penultimate['plural'].sum())
print(results['plural'].sum())
