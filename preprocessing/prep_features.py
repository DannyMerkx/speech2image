# -*- coding: utf-8 -*-
"""
Spyder Editor

Make a dictionary that pairs all the flickr8k images with their 5 captions

"""

import numpy as np
import tables
import os
import tables

from vgg16 import vgg
from audio_features import audio_features

# path to the flickr audio and image files
audio_path = os.path.join('/home/danny/Documents/Flickr/flickr_audio/wavs')
#audio_path = os.path.join('C:\\','Users', 'Beheerder','Documents','PhD','Flickr','flickr_audio', 'wavs')

img_path = os.path.join('/home/danny/Documents/Flickr/Flickr8k_Dataset/Flicker8k_Dataset')
#img_path = os.path.join('C:\\','Users', 'Beheerder','Documents','PhD','Flickr','Flickr8k_Dataset','Flicker8k_Dataset')
# list the img and audio directories
audio = os.listdir(audio_path)
imgs = os.listdir(img_path)
# strip the files to their basename and remove file extension
imgs_base = [x.split('.')[0] for x in imgs]
audio_base = [x.split('.')[0] for x in audio]

# match all image files with the appropriate audio files.
img_audio = {}
no_cap = []
for x in imgs_base:
    temp = []
    for y in audio_base:
        if x == y[:-2]:
           temp.append(y + '.wav')
        if len (temp) == 5:
            break
    if temp:
        img_audio[x + '.jpg' ] = temp
    else:
        # keep track of images without captions
        no_cap.append(x)
# create h5 output file for preprocessed images and audio
output_file = tables.open_file('/home/danny/Documents/Flickr/flickr_features.h5', mode='a')

#output_file = tables.open_file(os.path.join('C:\\','Users', 'Beheerder','Documents','PhD','Flickr','features.h5'), mode='a')
# we need to append something to the flickr files names because pytable group names cannot start
# with integers.
append_name = 'flickr_'

# create the h5 file to hold all image and audio features
for x in img_audio:
    # one group for each image file which will contain its vgg16 features and audio captions 
    output_file.create_group("/", append_name + x.split('.')[0])    
    
# create the vgg16 features for all images  
vgg(img_path, output_file, append_name)

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

audio_features(params, img_audio, audio_path, append_name)

# close the output files
output_file.close()
