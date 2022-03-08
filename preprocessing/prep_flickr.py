#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:49:51 2018

@author: danny
"""
import os
import json
import tables

from visual_features import vis_feats
from audio_features import audio_features
from text_features import text_features_flickr
from pathlib import Path

import numpy as np

# path to the flickr audio, caption and image files 
audio_path = os.path.join('/data/databases/flickr/flickr_audio/wavs')
img_path = os.path.join('/data/databases/flickr/Flicker8k_Dataset')
text_path = os.path.join('/data/databases/flickr/dataset.json')
# save the resulting feature file here
data_loc = '/vol/tensusers3/dmerkx/flickr_features.h5'
vis = ['resnet']
speech = ['fbanks', 'mfcc']
text = True

# list the img and audio directories
audio = os.listdir(audio_path)
imgs = os.listdir(img_path)

# strip the files to their basename and remove file extension
imgs_base = [x.split('.')[0] for x in imgs]
audio_base = [x.split('.')[0] for x in audio]

# create a dictionary with the common part in the images and audio filenames as 
# keys pointing to the image and caption file names
img_audio = {}
no_cap = []
for im in imgs_base:
    temp = []
    for cap in audio_base:
        # match the images with the appropriate captions
        if im in cap:
           temp.append(cap + '.wav')
        # quit the loop early if all captions are found
        if len (temp) == 5:
            break
    # make a dictionary entry with the basename of the files as key, containing
    # the full names of the img and audio files
    if temp:
        img_audio[im] = im + '.jpg' , temp
    else:
        # keep track of images without captions
        no_cap.append(im)
# we need to append something to the flickr files names because pytable group 
# names cannot start with integers.
append_name = 'flickr_'

# if the output file does not exist yet, create it
if not Path(data_loc).is_file():
    # create h5 output file for preprocessed images and audio
    output_file = tables.open_file(data_loc, mode='a')    
    for x in img_audio:
        try:        
            output_file.create_group("/", append_name + x.split('.')[0])    
        except:
            continue
# else load an existing file to append new features to      
else:
    output_file = tables.open_file(data_loc, mode='a')
    
#list all the nodes
node_list = output_file.root._f_list_nodes()
    
# create the visual features for all images
for ftype in vis: 
    vis_feats(img_path, output_file, append_name, img_audio, node_list, ftype) 

######### parameter settings for the audio preprocessing ###############
# put paramaters in a dictionary
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
params['normalise'] = True
#############################################################################

# create the audio features for all captions
for ftype in speech:
    params['feat'] = ftype
    audio_features(params, img_audio, audio_path, append_name, node_list)

# load all the captions
text_dict = {}
txt = json.load(open(text_path))['images']
for x in txt:
    text_dict[x['filename'].split('.')[0]] = x
# add text features for all captions
if text:
    text_features_flickr(text_dict, output_file, append_name, node_list)
# close the output files
output_file.close()
