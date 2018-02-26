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

from vgg16 import vgg
from audio_features import audio_features

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

# path to the flickr audio and image files
audio_path = os.path.join('/home/danny/Downloads/placesaudio_distro_part_1/')
#audio_path = os.path.join('C:\\','Users', 'Beheerder','Documents','PhD','Flickr','flickr_audio', 'wavs')

img_path = os.path.join('/home/danny/Documents/Flickr/Flickr8k_Dataset/Flicker8k_Dataset')
            
# for the places database, there is metadata availlable which links a unique 
# identifier to both the wavs and the images
meta_data_loc = '/home/danny/Downloads/placesaudio_distro_part_1/metadata/'

file = open(os.path.join(meta_data_loc, 'utt2wav'))

wavs = file.readlines()

file.close()

file = open(os.path.join(meta_data_loc, 'utt2image'))

imgs =file.readlines()

file.close()

# create a dictionary with the unique identifier as key pointing to the image and 
# its caption. The id contains a hyphen which is invalid for h5 naming conventions so
# it is replaced by an underscore
img_audio = {}
for im in imgs:
    im  = im.split()
    img_audio[im[0].replace('-','_')] = im[1]
    
for wav in wavs:
    wav = wav.split()
    img_audio[wav[0].replace('-','_')] = img_audio[wav[0].replace('-','_')], [wav[1]]
    
# create h5 output file for preprocessed images and audio
output_file = tables.open_file('/home/danny/Documents/data/places_features.h5', mode='a')

#output_file = tables.open_file(os.path.join('C:\\','Users', 'Beheerder','Documents','PhD','Flickr','features.h5'), mode='a')
# we need to append something to the flickr files names because pytable group names cannot start
# with integers.
append_name = 'places_'

# the size of this database is bigger than the maximum recommended amount of children
# a node can have so they cannot all be directly under the root node. instead the 
# database is split in smaller 10k bits.
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
vgg(img_path, output_file, append_name, img_audio, node_list) 

######### parameter settings for the audio preprocessing ###############

# option for which audio feature to create
feat = 'fbanks'
params = []
# set alpha for the preemphasis
alpha = 0.97
# set the number of desired filterbanks
nfilters = 40
# windowsize and shift in seconds
t_window = .025
t_shift = .010
# option to include delta and double delta features
use_deltas = False
# option to include frame energy
use_energy = False
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

audio_features(params, img_audio, audio_path, append_name, node_list)

# close the output files
output_file.close()
