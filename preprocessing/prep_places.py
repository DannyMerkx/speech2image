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

from visual_features import vis_feats
from audio_features import audio_features

vis = ['resnet', 'vgg19']
speech = ['fbanks', 'mfcc']

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
audio_path = os.path.join('/data/places_corpus/placesaudio_distro_part_1/')

img_path = os.path.join('/data/places_corpus/places_images')
            
# for the places database, there is metadata availlable which links a unique 
# identifier to both the wavs and the images
meta_data_loc = os.path.join(audio_path, 'metadata')

file = open(os.path.join(meta_data_loc, 'utt2wav'))

wavs = file.readlines()

file.close()

file = open(os.path.join(meta_data_loc, 'utt2image'))

imgs = file.readlines()

file.close()

# create a dictionary with the unique identifier as key pointing to the image 
# and its caption. The id contains a hyphen which is invalid for h5 naming 
# conventions so it is replaced by an underscore
img_audio = {}
for im in imgs:
    im  = im.split()
    img_audio[im[0].replace('-','_')] = im[1][1:]
    
for wav in wavs:
    wav = wav.split()
    img_audio[wav[0].replace('-','_')] = img_audio[wav[0].replace('-','_')], [wav[1]]
    
# create h5 output file for preprocessed images and audio
output_file = tables.open_file('/prep_data/places_features.h5', mode='a')


# we need to append something to the flickr files names because pytable group 
# names cannot start with integers.
append_name = 'places_'

# the size of this database is bigger than the maximum amount of children
# a single node can have. I split the database smaller 10k subgroups.
count = 0
for batch in batcher(10000, img_audio):
    node = output_file.create_group('/', 'subgroup_' + str(count))
    for x in batch:
        output_file.create_group(node, append_name + x.split('.')[0])
    count +=1

node_list = []
subgroups = output_file.root._f_list_nodes()
for subgroup in subgroups:
    node_list = node_list + subgroup._f_list_nodes()

# create the vgg16 features for all images  
for x in vis: 
    vis_feats(img_path, output_file, append_name, img_audio, node_list, x) 
######### parameter settings for the audio preprocessing ###############
# option for which audio feature to create (options are mfcc, fbanks, 
# freq_spectrum and raw)
feat = ''
params = []
# set alpha for the preemphasis
alpha = 0.97
# set the number of desired filterbanks
nfilters = 40
# windowsize and shift in seconds
t_window = .025
t_shift = .010
# option to include delta and double delta features
use_deltas = True
# option to include frame energy
use_energy = True
# put paramaters in a list
params.append(alpha)
params.append(nfilters) 
params.append(t_window)
params.append(t_shift)
params.append(feat)
params.append(output_file)
params.append(use_deltas)
params.append(use_energy)
#############################################################################

# create the audio features for all captions
for ftype in speech:
    params[4] = x
    audio_features(params, img_audio, audio_path, append_name, node_list)


# close the output files
output_file.close()
