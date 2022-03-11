#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:19:52 2019

@author: danny
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
text = []
with open('/home/danny/Downloads/mbn.out') as file:
    for line in file:
        if 'True' in line or 'False' in line:
            text.append(eval(line[line.find('['):line.find(']')+1]))


clean_text = [] 
for x in range(0, len(text), 4):
    clean_text.append(text[x:x+4])
        
full = []
for x in range(0, len(clean_text), 5):
    full.append(clean_text[x])
untrained = []
for x in range(1, len(clean_text), 5):
    untrained.append(clean_text[x])
single = [] 
for x in range(2, len(clean_text), 5):
    single.append(clean_text[x])
two = []
for x in range(3, len(clean_text), 5):
    two.append(clean_text[x])
three = []
for x in range(4, len(clean_text), 5):
    three.append(clean_text[x])
    
def TPR(TP,FN):
    return [TP[i]/(TP[i]+FN[i] + 1e-6) for i in range(len(TP)) ]
def TNR(TN,FP):
    return [TN[i]/(TN[i]+FP[i] + 1e-6) for i in range(len(TN)) ]
def FPR (TN,FP):
    return [FP[i]/(TN[i]+FP[i] + 1e-6) for i in range(len(TN)) ]
def precision(TP, FP):
    return [TP[i]/(FP[i]+TP[i] + 1e-6) for i in range(len(FP)) ]

def F1(TPR, recall):
    return [2* ((x * y) / (x + y + 1e-6)) for x,y in zip(TPR,recall)]   

n = 32

TPR_full = TPR(full[n][0], full[n][3])

TPR_untrained = TPR(untrained[n][0], untrained[n][3])

TPR_single = TPR(single[n][0], single[n][3])

TPR_two = TPR(two[n][0], two[n][3])

TPR_three = TPR(three[n][0], three[n][3])

##################
TNR_full = TNR(full[n][1], full[n][2])

TNR_untrained = TNR(untrained[n][1], untrained[n][2])

TNR_single = TNR(single[n][1], single[n][2])

TNR_two = TNR(two[n][1], two[n][2])

TNR_three = TNR(three[n][1], three[n][2])

####
FPR_full = FPR(full[n][1], full[n][2])

FPR_untrained = FPR(untrained[n][1], untrained[n][2])

FPR_single = FPR(single[n][1], single[n][2])

FPR_two = FPR(two[n][1], two[n][2])

FPR_three = FPR(three[n][1], three[n][2])
#####
PR_full = precision(full[n][0], full[n][2])

PR_untrained = precision(untrained[n][0], untrained[n][2])

PR_single = precision(single[n][0], single[n][2])

PR_two = precision(two[n][0], two[n][2])

PR_three = precision(three[n][0], three[n][2])

plt.plot(TPR_untrained, PR_untrained, marker = 'x', label = 'input_features')
plt.plot(TPR_single, PR_single, marker = 'x', label = '1st layer')
plt.plot(TPR_two, PR_two, marker = 'x', label = '2nd layer')
plt.plot(TPR_three, PR_three, marker = 'x', label = '3rd layer')
plt.plot(TPR_full, PR_full, marker = 'x', label = 'final layer (attention)')
plt.ylabel('precision')
plt.xlabel('recall')
plt.legend()
plt.show()

plt.plot(FPR_untrained, TPR_untrained, marker = 'x', label = 'input_features')
plt.plot(FPR_single, TPR_single, marker = 'x', label = '1st layer')
plt.plot(FPR_two, TPR_two, marker = 'x', label = '2nd layer')
plt.plot(FPR_three, TPR_three, marker = 'x', label = '3rd layer')
plt.plot(FPR_full, TPR_full, marker = 'x', label = 'final layer (attention)')
plt.ylabel('recall')
plt.xlabel('False positive rate')
plt.legend()
plt.show()

plt.plot(F1(TPR_untrained, PR_untrained), marker = 'x', label = 'input_features')
plt.plot(F1(TPR_single, PR_single), marker = 'o', label = '1st layer')
plt.plot(F1(TPR_two, PR_two), marker = '^', label = '2nd layer')
plt.plot(F1(TPR_three, PR_three), marker = 'd', label = '3rd layer')
plt.plot(F1(TPR_full, PR_full), marker = 's', label = 'final layer (attention)')
plt.ylabel('F1 score') 
plt.xlabel('detection threshold')
plt.xticks([x for x in range(0,21,2)], np.linspace(0,1,11).round(1)) 
plt.legend()
plt.show()


plt.plot(TPR_two, marker = 'x', label = 'sensitivity')
plt.plot(TNR_two, marker = 'x', label = 'specificity')
plt.xticks([x for x in range(0,21,2)], np.linspace(0,1,11).round(1)) 
plt.legend()
plt.show()