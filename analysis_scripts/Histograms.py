#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:02:30 2021
Create the histograms for use in the paper 
@author: danny
"""
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from patn_test import evaluator


######################## word recognition ################################  
data_loc = './test_features.h5'
# word types can be tested seperately just in case
word_types = ['singular', 'plural', 'root', 'third', 'participle']   

cap_model = '../models/caption_model.16'
img_model = '../models/image_model.16'

patn_evaluator = evaluator(f_loc = data_loc, img_loc = img_model, 
                           cap_loc = cap_model, enc_type = 'rnn')

patn_evaluator.recognition(word_types, no_gating = True, n = 10)
recognition = patn_evaluator.conv_rec_dict()

cap_model = '../models/caption_model.16VQ'
img_model = '../models/image_model.16VQ'

patn_evaluator_VQ = evaluator(f_loc = data_loc, img_loc = img_model, 
                              cap_loc = cap_model, enc_type = 'rnn')

patn_evaluator_VQ.recognition(word_types, no_gating = True, n = 10)
recognition_VQ = patn_evaluator.conv_rec_dict()

words = []
for d in recognition.keys():
    for key in recognition[d].keys():    
            words.append([recognition[d][key]/20, d])

x = pd.DataFrame(words)
x.columns = ['P@10', 'Word_type']
        
x['P@10_bin'] = pd.cut(x['P@10'], bins = [-0.1,.04,.14,.24,.34,.44,.54,.64,.74,.84,.94, 1],
                       labels = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]).astype(float)
fig, ax = plt.subplots(2,2,  figsize=[9,6], sharex = True)

sb.histplot(x = 'P@10_bin', hue ='Word_type', 
            data = x[(x['Word_type'] == 'plural') | (x['Word_type'] == 'singular')], 
            multiple='stack', bins = 11, ax = ax[0,0], legend=False)
ax[0,0].set_xticks(np.linspace(0.05, .96 ,11))
ax[0,0].set_xticklabels(['0','.1','.2','.3','.4','.5','.6','.7','.8','.9','1'])
ax[0,0].set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
ax[0,0].set_xlabel('')
ax[0,0].legend(title ='Noun type', labels = ['plural', 'singular'])
ax[0,0].set_title('RNN noun recognition')

sb.histplot(x = 'P@10_bin', hue ='Word_type', 
            data = x[(x['Word_type'] == 'root') | (x['Word_type'] == 'third') | (x['Word_type'] == 'participle')], 
            multiple='stack', bins = 11, ax = ax[0,1])
ax[0,1].set_xticks(np.linspace(0.05, .96 ,11))
ax[0,1].set_xticklabels(['0','.1','.2','.3','.4','.5','.6','.7','.8','.9','1'])
ax[0,1].set_xlabel('')
ax[0,1].legend(title ='Verb type', labels = ['participle', 'third', 'root'])
ax[0,1].set_title('RNN verb recognition')
ax[0,1].set_ylabel('')
ax[0,1].set_yticks([0,10,20,30,40, 50])


words = []
for d in recognition_VQ.keys():
    for key in recognition_VQ[d].keys():    
            words.append([recognition_VQ[d][key]/20, d])

x = pd.DataFrame(words)
x.columns = ['P@10', 'Word_type']
        
x['P@10_bin'] = pd.cut(x['P@10'], bins = [-0.1,.04,.14,.24,.34,.44,.54,.64,.74,.84,.94, 1], 
                       labels = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9, 1]).astype(float)

sb.histplot(x = 'P@10_bin', hue ='Word_type', 
            data = x[(x['Word_type'] == 'plural') | (x['Word_type'] == 'singular')], 
            multiple='stack', bins = 11, ax = ax[1,0], legend=False)
ax[1,0].set_xticks(np.linspace(0.05, .96 ,11))
ax[1,0].set_xticklabels(['0','.1','.2','.3','.4','.5','.6','.7','.8','.9','1'])
ax[1,0].set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
ax[1,0].set_xlabel('P@10')
ax[1,0].set_title('RNN-VQ noun recognition')

x['P@10_bin'] = pd.cut(x['P@10'], bins = [-0.1,.04,.14,.24,.34,.44,.54,.64,.74,.84,.94, 1], 
                       labels = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]).astype(float)

sb.histplot(x = 'P@10_bin', hue ='Word_type', 
            data = x[(x['Word_type'] == 'root') | (x['Word_type'] == 'third') | (x['Word_type'] == 'participle')], 
            multiple='stack', bins = 10, ax = ax[1,1], legend = False)
ax[1,1].set_xticks(np.linspace(0.05, .96 ,11))
ax[1,1].set_xticklabels(['0','.1','.2','.3','.4','.5','.6','.7','.8','.9','1'])
ax[1,1].set_xlabel('P@10')
ax[1,1].set_title('RNN-VQ verb recognition')
ax[1,1].set_ylabel('')
ax[1,1].set_yticks([0,10,20,30,40])

plt.gcf().subplots_adjust(bottom=0.15)

plt.savefig(fname = '/home/danny/Pictures/recog_hist.png', dpi = 1000)


