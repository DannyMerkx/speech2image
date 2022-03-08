#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:55:37 2022

@author: danny
"""
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

results = tester.results
datasets = [x for x in results.keys()]
models = [x for x in results['MEN'].keys() if not 'partial' in x]
partials = [x for x in results['MEN'].keys() if 'partial' in x]

df = pd.DataFrame(columns = ['Model', 'Task', 'R^2', 'Partial', 'pretrained'])

for x in results.keys():
    for y in results[x].keys():
        partial = 'partial' in y
        if not partial:
            model = y
            try:
                partial = results[x][f'{y}_partial']**2
            except:
                partial = 0
            R2 = results[x][y]**2
            pretrained = 'pretrained' in model
            row = [model, x, R2, partial + R2, pretrained]
            df.loc[len(df.index)] = row

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

fig, ax  = plt.subplots(1,2, figsize = (15,5))
# plot the sum of R2 + partial R2
sb.barplot(data = df[df['pretrained'] == False], x= 'Task', y = 'Partial',
           hue = 'Model', ci = None, ax = ax[0], palette = [[0.4,0.4,0.4],
                                                            [0,0,0],
                                                            [0.2,0.2,0.2]
                                                            ]
           )
# plot R2
sb.barplot(data = df[df['pretrained'] == False], x= 'Task', y = 'R^2',
           hue = 'Model', ci = None, ax = ax[0], palette = ['blue', 'orange', 
                                                            'green', 'red']
           )
# same for pretrained models
sb.barplot(data = df[df['pretrained'] == True], x= 'Task', y = 'Partial', 
           hue = 'Model', ci = None, ax = ax[1], palette = [[0,0,0],
                                                            [0.2,0.2,0.2],
                                                            [0.4,0.4,0.4]
                                                            ]
           )
sb.barplot(data = df[df['pretrained'] == True], x= 'Task', y = 'R^2', 
           hue = 'Model', ci = None, ax = ax[1], palette = ['orange', 'green', 
                                                            'red']
           )
# the subplots can share a y label
ax[0].set_ylabel('$R^{2}$')
ax[1].set_ylabel('')
ax[0].set_xlabel('Dataset')
ax[1].set_xlabel('Dataset')
ax[0].set_yticks([0.1,0.2,0.3,0.4,0.5,0.6])
ax[1].set_yticks([0.15,0.3,0.45,0.6,0.75,0.9])
# rotate the x labels
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 'vertical')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 'vertical')

# remove the handles from the greyscale plot from the legend and remove the 
# legend from subplot 2
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles[4:], labels[4:])
ax[1].legend().remove()
# set the appropriate titles
ax[0].set_title('MSCOCO models')
ax[1].set_title('Pretrained models')
plt.savefig('rsquared.png', dpi= 600, bbox_inches = 'tight')
plt.show()
