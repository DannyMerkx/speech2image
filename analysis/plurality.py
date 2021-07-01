#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 16:41:01 2021

@author: danny
"""
import pandas as pd

noun_annot = pd.read_csv(open('../Image_annotations/annotations/Noun_annotations.csv'), 
                          index_col=0)
verb_annot = pd.read_csv(open('../Image_annotations/annotations/Verb_annotations.csv'), 
                          index_col=0)

data_loc = './lmer.csv'

results = pd.read_csv(data_loc, index_col=0)

#results = results[results['full_word']]
results = results[results['gate'] == results['n_gates']-1]
results = results[(results['s_id'] == 's1_1') | (results['s_id'] == 's2_1')]

results = results[results['VQ'] == 0]

# get only those nouns which had at least 10 images with plural annotation
plural_nouns = (noun_annot[noun_annot == 2].sum(0) > 9) & (noun_annot[noun_annot == 1].sum(0) > 9)
# drop shorst and sunglasses as those have no plural/singular target word
plural_nouns = plural_nouns.drop('shorts')
plural_nouns = plural_nouns.drop('sunglasses')
plural_nouns = [x for x in plural_nouns.index if plural_nouns[x]]
mapping = {'dogs': 'dog', 'men': 'man', 'boys': 'boy', 'girls': 'girl',
            'women': 'woman', 'shirts': 'shirt', 'balls': 'ball',
            'groups': 'group', 'rocks': 'rock', 'cameras': 'camera', 
            'bikes': 'bike', 'mountains': 'mountain', 'hats': 'hat', 
            'players': 'player', 'jackets': 'jacket', 'cars': 'car', 
            'buildings': 'building', 'dresses': 'dress', 'tables': 'table',
            'hands': 'hand', 'trees': 'tree', 'hills': 'hill', 
            'toys': 'toy', 'babies': 'baby', 'waves': 'wave',
            'benches': 'bench', 'sticks': 'stick', 'teams': 'team'}
# get recognition scores for singular and plural targets
plural = results[results['word_form'] == 'plural']
singular = results[results['word_form'] == 'singular']

pl_filtered = {}
si_filtered = {}
for key in plural['word']:
    w = key
    if key in mapping.keys():
        w = mapping[key]
    C = plural[plural['word'] == key]
    if w in plural_nouns:
        pl_filtered[key] = [(C['p@10']-C['plural']).sum(), C['plural'].sum()]

for key in singular['word']:
    C = singular[singular['word'] == key]
    if key in plural_nouns:
        si_filtered[key] = [(C['p@10']-C['plural']).sum(), C['plural'].sum()]

pl_filtered = pd.DataFrame(pl_filtered).transpose()
si_filtered = pd.DataFrame(si_filtered).transpose()        

print(pl_filtered.sum(0)/pl_filtered.sum(0).sum())
print(pl_filtered.sum(0))
print(si_filtered.sum(0)/si_filtered.sum(0).sum())
print(si_filtered.sum(0))
