#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:54:47 2021

@author: danny
"""
from collections import defaultdict
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def group_gating(gating_dict):
    grouped = defaultdict(list)
    for key in gating_dict.keys():
        k = key.split('_')[0]
        grouped[k].append(gating_dict[key])
    return grouped

words = gating_recognition['singular']

grouped = group_gating(words)
l = [len(grouped[x]) for x in grouped.keys()]

for x in grouped.keys():
    while len(grouped[x]) < max(l):
        grouped[x].append(np.nan)
        
data = pd.DataFrame(grouped).transpose()
data['len'] = l
data = data.sort_values('len')
fig, ax = plt.subplots(2,2, figsize=[10,15], sharex = 'col')


#sb.heatmap(data= data.iloc[:, :-1], ax = ax, cmap='crest', annot = True)

sb.heatmap(data= data.iloc[:, :-1]/20, ax = ax[0,0], cmap='crest',
           cbar=False, yticklabels = list(data.index))
ax[0,0].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5])
ax[0,0].set_xticklabels(['1','2','3','4','5','6','7','8','9'])
ax[0,0].set_title('RNN singular noun gating')

#plt.savefig(fname = '/home/danny/Pictures/sing_hm.png', dpi = 1000)

words = gating_recognition['plural']

grouped = group_gating(words)
l = [len(grouped[x]) for x in grouped.keys()]

for x in grouped.keys():
    while len(grouped[x]) < max(l):
        grouped[x].append(np.nan)
        
data = pd.DataFrame(grouped).transpose()
data['len'] = l
data = data.sort_values('len')

#sb.heatmap(data= data.iloc[:, :-1], ax = ax, cmap='crest', annot = True)

sb.heatmap(data= data.iloc[:, :-1]/20, ax = ax[0,1], cmap='crest',
           cbar= False, yticklabels = list(data.index))
ax[0,1].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5])
ax[0,1].set_xticklabels(['1','2','3','4','5','6','7','8','9', '10'])
ax[0,1].set_title('RNN plural noun gating')

############################################ vq ##############################
words = gating_recognition_VQ['singular']

grouped = group_gating(words)
l = [len(grouped[x]) for x in grouped.keys()]

for x in grouped.keys():
    while len(grouped[x]) < max(l):
        grouped[x].append(np.nan)
        
data = pd.DataFrame(grouped).transpose()
data['len'] = l
data = data.sort_values('len')

#sb.heatmap(data= data.iloc[:, :-1], ax = ax, cmap='crest', annot = True)

sb.heatmap(data= data.iloc[:, :-1]/20, ax = ax[1,0], cmap='crest', cbar=False,
           yticklabels = list(data.index))
ax[1,0].set_xlabel('No. phonemes seen')
ax[1,0].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5])
ax[1,0].set_xticklabels(['1','2','3','4','5','6','7','8','9'])
ax[1,0].set_title('RNN VQ singular noun gating')

words = gating_recognition_VQ['plural']
grouped = group_gating(words)
l = [len(grouped[x]) for x in grouped.keys()]

for x in grouped.keys():
    while len(grouped[x]) < max(l):
        grouped[x].append(np.nan)
        
data = pd.DataFrame(grouped).transpose()
data['len'] = l
data = data.sort_values('len')
#sb.heatmap(data= data.iloc[:, :-1], ax = ax, cmap='crest', annot = True)
cbar_ax = fig.add_axes([.9, .09, .03, .4])

sb.heatmap(data= data.iloc[:, :-1]/20, ax = ax[1,1], cmap='crest', 
           cbar=True, cbar_ax= cbar_ax,yticklabels = list(data.index),
           cbar_kws={'label': 'P@10'})
ax[1,1].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5])
ax[1,1].set_xticklabels(['1','2','3','4','5','6','7','8','9','10'])
ax[1,1].set_xlabel('No. phonemes seen')
ax[1,1].set_title('RNN VQ plural noun gating')
plt.tight_layout()
plt.savefig(fname = '/home/danny/Pictures/gating_2.png', dpi = 200)


def avg_grouping(grouped):
    # average the p@10 for words of the same length
    lens = {len(grouped[w]):[np.zeros(len(grouped[w])),0] for w in grouped}
    for x in lens.keys():
        for w in grouped.keys():
            if x == len(grouped[w]):
                lens[x][0] += grouped[w]
                lens[x][1] += 1
    return [lens[x][0]/(lens[x][1]*20) for x in lens.keys()]

############################# line plot of averaged gating ####################
grs = avg_grouping(group_gating(gating_recognition['singular']))
grp = avg_grouping(group_gating(gating_recognition['plural']))

fig, ax = plt.subplots(2,2, sharex = 'col', sharey = True)
grs = np.array(grs)[np.argsort([len(x) for x in grs])]
grs_legend = [2, 18, 16, 6, 3, 2, 2]
grp = np.array(grp)[np.argsort([len(x) for x in grp])]
grp_legend = [3, 16, 11, 6, 3, 2, 1]

for x in grs:
    sb.lineplot(data = x, ax = ax[0,0])
    ax[0,0].set_xticks([0,1,2,3,4,5,6,7,8])
    ax[0,0].set_yticks([0,.2,.4,.6,.8,1])
    ax[0,0].set_ylabel('Average P@10')
    ax[0,0].set_title('LSTM singular')
    #ax[0].legend(labels = grs_legend)
for x in grp:
    sb.lineplot(data = x, ax = ax[0,1])
    ax[0,1].set_xticks([0,1,2,3,4,5,6,7,8,9])
    ax[0,1].set_yticks([0,.2,.4,.6,.8,1])
    ax[0,1].set_title('LSTM plural')
 
grs = avg_grouping(group_gating(gating_recognition_VQ['singular']))
grp = avg_grouping(group_gating(gating_recognition_VQ['plural']))

grs = np.array(grs)[np.argsort([len(x) for x in grs])]
grs_legend = [2, 18, 16, 6, 3, 2, 2]
grp = np.array(grp)[np.argsort([len(x) for x in grp])]
grp_legend = [3, 16, 11, 6, 3, 2, 1]

for x in grs:
    sb.lineplot(data = x, ax = ax[1,0])
    ax[1,0].set_xticks([0,1,2,3,4,5,6,7,8])
    ax[1,0].set_xticklabels(['1','2','3','4','5','6','7','8','9'])
    ax[1,0].set_yticks([0,.2,.4,.6,.8,1])
    ax[1,0].set_ylabel('Average P@10')
    ax[1,0].set_xlabel('No. of phonemes')
    ax[1,0].set_title('LSTM VQ singular')
    #ax[0].legend(labels = grs_legend)
for x in grp:
    sb.lineplot(data = x, ax = ax[1,1])
    ax[1,1].set_xticks([0,1,2,3,4,5,6,7,8,9])
    ax[1,1].set_xticklabels(['1','2','3','4','5','6','7','8','9','10'])
    ax[1,1].set_xlabel('No. of phonemes')
    ax[1,1].set_yticks([0,.2,.4,.6,.8,1])
    ax[1,1].set_title('LSTM VQ plural')   
plt.tight_layout()
plt.savefig(fname = '/home/danny/Pictures/gating_lineplot.png', dpi = 200)
