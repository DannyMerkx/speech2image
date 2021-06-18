#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:31:13 2018
Add multilingual bottleneck features to existing flickr8k features file
@author: danny
"""
import numpy as np
import tables
import re

f_atom= tables.Float32Atom()
mbn_file = np.load('/data/mbn.npz')
feature_file = tables.open_file('/prep_data/flickr_features.h5', mode = 'a')

d = dict(zip((k for k in mbn_file), (mbn_file[k] for k in mbn_file)))

keys = [x for x in d.keys()]

nodes = feature_file.root._f_list_nodes()
count = 1
c = dict()

regex_1 = re.compile('[0-9a-z]+_[0-9a-z]+_[0-5]')
regex_2 = re.compile('[0-9a-z]+_[0-9a-z]+')

for x in nodes:
    audio_node = feature_file.create_group(x, 'mbn') 
    for y in keys:
        
        if regex_2.search(y).group() in x._v_name:
            print('processing file:' + str(count))            
            count += 1
            mbn_features = d[y] 
            feature_shape= np.shape(mbn_features)[1]
            f_table = feature_file.create_earray(audio_node, 'flickr_' + regex_1.search(y).group(), f_atom, (0,feature_shape),expectedrows=5000)
        
            # append new data to the tables
            f_table.append(mbn_features)

