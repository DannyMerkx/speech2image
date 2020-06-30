#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:03:28 2018
prepare the places database

@author: danny
"""


import numpy as np
import os
import tables
import json

from visual_features import vis_feats
from audio_features import audio_features
from pathlib import Path

# places is a rather large dataset, be prepared for 20-60GB of extra data just
# by adding another feature type. vis features can be resnet, vgg19,
# resnet_trunc or raw
vis = ['resnet']
# speech features can be raw, freq_spectrum, fbanks or mfcc
speech = ['mfcc']

data_loc = '/vol/tensusers3/dmerkx/places_features.h5'

def batcher(batch_size, img_audio):
    keys = [x for x in img_audio]
    for start_idx in range(0, len(img_audio) - 1, batch_size):
        excerpt = {}
        if not start_idx + batch_size > len(img_audio):
            for x in range(start_idx, start_idx + batch_size):
                excerpt[keys[x]] = img_audio[keys[x]]
            yield excerpt
        else:
            for x in range(start_idx, len (img_audio)):
                excerpt[keys[x]] = img_audio[keys[x]]
            yield excerpt

# path to the audio and image files
audio_path = os.path.join('/vol/tensusers3/dmerkx/databases/places/PlacesAudio_400k_distro/')

img_path = os.path.join('/vol/tensusers3/dmerkx/databases/places/img_data')

meta_data_loc = '/vol/tensusers3/dmerkx/databases/places/'
val = json.load(open(os.path.join(meta_data_loc, 'val.json')))
train = json.load(open(os.path.join(meta_data_loc, 'train.json')))

# create a dictionary with the unique identifier as key pointing to the image 
# and its caption. The id contains a hyphen which is invalid for h5 naming 
# conventions so it is replaced by an underscore
img_audio = {}
for im in train['data']:
    img_audio[im['uttid'].replace('-', '_')] = im['image'], [im['wav']]
for im in val['data']:
    img_audio[im['uttid'].replace('-', '_')] = im['image'], [im['wav']]

    

# we need to append something to the flickr files names because pytable group 
# names cannot start with integers.
append_name = 'places_'

if not Path(data_loc).is_file():
    # create h5 output file for preprocessed images and audio
    output_file = tables.open_file(data_loc, mode='a')
    count = 0
    # the size of this database is bigger than the maximum amount of children
    # a single node can have. I split the database smaller 10k subgroups.
    for batch in batcher(10000, img_audio):
        node = output_file.create_group('/', 'subgroup_' + str(count))
        for x in batch:
            output_file.create_group(node, append_name + x)
        count +=1
else:
    # create h5 output file for preprocessed images and audio
    output_file = tables.open_file(data_loc, mode='a')

node_list = []
subgroups = output_file.root._f_list_nodes()
for subgroup in subgroups:
    node_list = node_list + subgroup._f_list_nodes()

# create the visual features for all images  
for ftype in vis:
    vis_feats(img_path, output_file, append_name, img_audio, node_list, ftype)
######### parameter settings for the audio preprocessing ###############
# put the parameters in a dictionary
params = {}
params['alpha'] = 0.97
params['nfilters'] = 40
params['ncep'] = 13
params['t_window'] = .025
params['t_shift'] = 0.01
params['feat'] = ''
params['output_file'] = output_file
params['use_deltas'] = True
params['use_energy'] = True
params['windowing'] = np.hamming
params['delta_n'] = 2
#############################################################################

# create the audio features for all captions
for ftype in speech:
    params['feat'] = ftype
    audio_features(params, img_audio, audio_path, append_name, node_list)

# close the output files
output_file.close()
