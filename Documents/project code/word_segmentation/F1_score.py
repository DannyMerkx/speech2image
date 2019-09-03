#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:13:30 2019

@author: danny
"""
import numpy as np

def F1_score(gs, segs):
    batch_size = len(segs)
    segs = [[0] + list(np.cumsum(segs[i])) for i in range(batch_size)]
    
    segs = [[[s[x - 1], s[x]] for x in range(1, len(s))] for s in segs]    

    gs_segs = [[0] + list(np.cumsum([len(x) for x in y.split()])) for y in gs]
    gs_segs = [[[s[x - 1], s[x]] for x in range(1, len(s))] for s in gs_segs]    
    
    
    tp = 0
    fp = 0

    fn = 0
    for gs_seg, seg in zip(gs_segs, segs):
        
        for gs in gs_seg:
            if gs in seg:
                tp += 1
            else:
                fn += 1
        for s in seg:
            if not s in gs_seg:
                fp += 1
                
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)    
    F1 = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, F1