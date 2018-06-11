#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:49:51 2018

@author: danny
"""
import tables
import os
import json
import pickle
from visual_features import vis_feats
from audio_features import audio_features
from text_features import text_features_flickr
# path to the flickr audio, caption and image files 
audio_path = os.path.join('/data/flickr/flickr_audio/wavs')
img_path = os.path.join('/data/flickr/Flickr8k_Dataset/Flicker8k_Dataset')
text_path = os.path.join('/data/speech2image/preprocessing/dataset.json')
# save the resulting feature file here
data_loc = os.path.join('/prep_data/flickr_features.h5')
# path to the word frequency dictionary
freq_dict_loc = os.path.join('/data/speech2image/preprocessing/dictionaries/flickr_frequency') 
# some bools in case only some new features needs to be added
vis = True
speech = True
text = True

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
# load the word frequency dictionary    
freq_dict = load_obj(freq_dict_loc)

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

# create h5 output file for preprocessed images and audio
output_file = tables.open_file(data_loc, mode='a')

# we need to append something to the flickr files names because pytable group names cannot start
# with integers.
append_name = 'flickr_'

# create the h5 file to hold all image and audio features. This will fail if they already excist such
# as when you run this file to append new features to an excisting feature file
for x in img_audio:
    try:        
        # one group for each image file which will contain its vgg16 features and audio captions 
        output_file.create_group("/", append_name + x.split('.')[0])    
    except:
        continue
#list all the nodes
node_list = output_file.root._f_list_nodes()
    
# create the visual features for all images 
if vis: 
    vis_feats(img_path, output_file, append_name, img_audio, node_list, 'resnet') 

######### parameter settings for the audio preprocessing ###############

# option for which audio feature to create (options are mfcc, fbanks, freq_spectrum and raw)
feat = 'mfcc'
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
if speech:
    audio_features(params, img_audio, audio_path, append_name, node_list)

# load all the captions
text_dict = {}
txt = json.load(open(text_path))['images']
for x in txt:
    text_dict[x['filename'].split('.')[0]] = x
# add text features for all captions
if text:
    text_features_flickr(text_dict, output_file, append_name, node_list, freq_dict)
# close the output files
output_file.close()
