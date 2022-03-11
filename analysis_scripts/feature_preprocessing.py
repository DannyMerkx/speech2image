#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:40:16 2021

This script prepares the test data by extracting image features from the
flickr test images and mfccs/filterbanks from the recorded test utterances
and storing them in an h5 file
@author: danny
"""
import os
import glob
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import PIL.Image
import tables
import librosa

import pandas as pd
import numpy as np
import python_speech_features.base as base
import python_speech_features.sigproc as sigproc

from collections import defaultdict
from pathlib import Path
# resnet model minus the penultimate classificaton layer
def resnet():
    model = models.resnet152(pretrained = True)
    model = nn.Sequential(*list(model.children())[:-1])
    return model

# resize and take ten crops of the image. Return the average activations over 
# the crops
def prep_tencrop(im, model): 
    # takes ten crops of 224x224 pixels
    tencrop = transforms.TenCrop(224)
    tens = transforms.ToTensor()
    # normalises the images using the values provided with torchvision
    normalise = transforms.Normalize(mean = [0.485,0.456,0.406], 
                                     std = [0.229, 0.224, 0.225])
    resize = transforms.Resize(256, PIL.Image.ANTIALIAS)
    
    im = tencrop(resize(im))
    im = torch.cat([normalise(tens(x)).unsqueeze(0) for x in im])
    if torch.cuda.is_available():
        im = im.cuda()
    activations = model(im)
    return activations.mean(0).squeeze()

def vis_feats(img_path, images):
    feature_dict = defaultdict(int)
    # prepare the pretrained model
    model = resnet()
    get_activations = prep_tencrop
    
    # set the model to use cuda
    if torch.cuda.is_available() and model:
        model = model.cuda()
    # disable gradients and dropout
    for p in model.parameters():
    	p.requires_grad = False
    model.eval()
    
    for img in images:
        #print(f'processing file: i')
        img_file = img_path + img
        
        im = PIL.Image.open(img_file).convert('RGB')
        im.requires_grad = False
        activations = get_activations(im, model).data.cpu().numpy()
        feature_dict[img] = activations
    return(feature_dict)

def audio_feats(audio_files, params):
    audio_features = {}

    for audio in audio_files:         
        input_data, fs = librosa.load(audio, sr = None)      
        # set the fft size to the power of two equal to or greater than 
        # the window size.
        #input_data = (input_data - input_data.mean())/input_data.std()
        window_size = int(fs * params['t_window'])
        exp = 1
        while True:
            if np.power(2, exp) - window_size >= 0:
                fft_size = np.power(2, exp)
                break
            else:
                exp += 1

###############################################################################        
        # create audio features 
        if params['feat'] == 'raw':
            # calculate the needed frame shift, premphasize and frame
            # the signal
            frame_shift = int(fs * params['t_shift'])
            input = sigproc.preemphasis(input_data, 
                                        coeff = params['alpha'])
            features = sigproc.framesig(input_data, 
                                        frame_len = window_size, 
                                        frame_step = frame_shift, 
                                        winfunc = params['windowing']
                                        )
    
        elif params['feat'] == 'freq_spectrum':
            # calculate the needed frame shift, premphasize and frame
            # the signal
            frame_shift = int(fs * params['t_shift'])
            input = sigproc.preemphasis(input_data, 
                                        coeff = params['alpha'])
            frames = sigproc.framesig(input, frame_len = window_size, 
                                      frame_step = frame_shift, 
                                      winfunc = params['windowing']
                                      )
            # create the power spectrum
            features = sigproc.powspec(frames, fft_size)
            
        elif params['feat'] == 'fbanks':
            # create mel filterbank features
            [features, energy] = base.fbank(input_data, samplerate = fs, 
                                            winlen = params['t_window'], 
                                            winstep = params['t_shift'], 
                                            nfilt = params['nfilters'], 
                                            nfft = fft_size, lowfreq = 0, 
                                            highfreq = None, 
                                            preemph = params['alpha'], 
                                            winfunc = params['windowing']
                                            )
        
        elif params['feat'] == 'mfcc':
            # create mfcc features
            features = base.mfcc(input_data, samplerate = fs,
                                 winlen = params['t_window'], 
                                 winstep = params['t_shift'], 
                                 numcep = params['ncep'], 
                                 nfilt = params['nfilters'], 
                                 nfft = fft_size, lowfreq = 0, 
                                 highfreq = None, 
                                 preemph = params['alpha'], ceplifter = 0, 
                                 appendEnergy = params['use_energy'], 
                                 winfunc = params['windowing']
                                 )
        # optionally add the deltas and double deltas
        features = (features - features.mean(0))/features.std(0)
        if params['use_deltas']:         
            single_delta = base.delta(features, params['delta_n'])
            double_delta = base.delta(single_delta, params['delta_n'])
            features= np.concatenate([features, single_delta, 
                                      double_delta], 1
                                     )
        audio_features[audio] = features
    return audio_features
###############################################################################               

img_path = os.path.join('../flickr_imgs/Flicker8k_Dataset/')
# read the image annotation files
noun_annot = pd.read_csv(open('../Image_annotations/annotations/Noun_annotations.csv'), 
                         index_col=0)
verb_annot = pd.read_csv(open('../Image_annotations/annotations/Verb_annotations.csv'), 
                         index_col=0)
# set a path for the processed data file
data_loc = os.path.join('./test_features.h5')
# from the annotation files get all the jpg names
image_files = list(noun_annot.index)

audio_script = open('../word_recordings/script.txt').read().split()
lemmas = open('../word_recordings/lemmas.txt').read().split()
word_types = open('../word_recordings/word_type.txt').read().split()
word_forms = open('../word_recordings/word_form.txt').read().split()

# list and sort the audio files
path = '../word_recordings/*/*'
audio = glob.glob(path)
audio_dict = defaultdict(list)
for a in audio:
    i = a.split('/')[-2]
    audio_dict[i].append(a)

for s in audio_dict.keys():
    _ = [int(x.split('/')[-1].split('-')[0]) for x in audio_dict[s]]
    audio_dict[s] = [audio_dict[s][x] for x in np.argsort(_)]
    
############################ alignment #######################################
alignment = open('../word_recordings/ali.1.ctm').read().split('\n')
# make sure fr is the same as the frame shift for the mfccs
fr = 0.01
phon_alignment = defaultdict(list)
# combine the alignments per word per speaker in a dictionary
for a in alignment:
    if not a == '':
        utt_id, channel_num, start_time, phone_dur, phone_id = a.split()
        idx = utt_id.split('_')[-1].split('.')[0]
        spk_id = f'{utt_id.split("_")[0]}_{utt_id.split("_")[1]}' 
        start_frame = float(start_time)/fr
        end_frame = start_frame + float(phone_dur)/fr
        phon_alignment[f'{spk_id}_{idx}'].append((int(start_frame), int(end_frame), phone_id))
# silence wont count as a phone for the experiment, add beginning and trailing 
# silence to the first and last phone
for key in phon_alignment.keys():
    a = phon_alignment[key]
    if a[0][-1] == '1':   
        x = [(a[0][0], a[1][1], a[1][2])]
        x.extend(a[2:])
        phon_alignment[key] = x
    a = phon_alignment[key]
    if a[-1][-1] == '1':
        x = [(a[-2][0], a[-1][1], a[-2][2])]
        y = a[:-2]
        y.extend(x)
        phon_alignment[key] = y
##############################################################################
# if the output file does not exist yet, create it
if not Path(data_loc).is_file():
    # create nodes to hold the audio and the image data
    output_file = tables.open_file(data_loc, mode='a')    
    output_file.create_group("/", 'image')
    output_file.create_group('/', 'audio')
    # each image gets its own subnode in image
    for x in image_files:
        try:        
            output_file.create_group("/image", 'img_' + x.split('.')[0])    
        except:
            continue
    for s in audio_dict.keys():
        for a in audio_dict[s]:
            output_file.create_group('/audio', f'{s}_{a.split("/")[-1].split(".")[0].replace("-", "_")}')
# else load an existing file to append new features to      
else:
    output_file = tables.open_file(data_loc, mode='a')
# list all the image and audio nodes
image_nodes = output_file.root.image._f_list_nodes()
audio_nodes = output_file.root.audio._f_list_nodes()
# process all the flickr test images
img_features = vis_feats(img_path, image_files)

# parameters for the audio features
params = {}
params['alpha'] = 0.97
params['nfilters'] = 40 
params['ncep'] = 13
params['t_window'] = .025
params['t_shift'] = 0.01
params['feat'] = 'mfcc'
params['output_file'] = output_file
params['use_deltas'] = True 
params['use_energy'] = True
params['windowing'] = np.hamming
params['delta_n'] = 2


audio_features = audio_feats([a for s in audio_dict.keys() for a in audio_dict[s]], 
                             params)
    
# create pytable atom for the features   
f_atom = tables.Float32Atom()   

for node in image_nodes:
    img_name = node._v_name[4:] +'.jpg'
    
    features = img_features[img_name]
    nouns = []
    for key in noun_annot.loc[img_name].to_dict():
        for x in range(noun_annot.loc[img_name].to_dict()[key]):
            nouns.append(key)
    verbs = []
    for key in verb_annot.loc[img_name].to_dict():
        if verb_annot.loc[img_name].to_dict()[key] > 0:
            verbs.append(key)
    
    output_file.create_array(node, 'resnet', features)
    output_file.create_array(node, 'nouns', nouns)
    output_file.create_array(node, 'verbs', verbs)


path = '../word_recordings/'
for node in audio_nodes:
    audio_name = node._v_name[5:].replace('_', '-') + '.wav'
    s_id = node._v_name[:4]
    
    audio_path = f'{path}{s_id}/'
    
    sent_id = int(audio_name.split('-')[0])
    ali = phon_alignment[f'{s_id}_{sent_id - 1}']
    transcript = audio_script[sent_id -1]
    lemma = lemmas[sent_id -1]
    word_type = word_types[sent_id -1]
    word_form = word_forms[sent_id -1]

    features = audio_features[audio_path + audio_name]
    # create new leaf node in the feature node for the current audio file
    feature_shape= np.shape(features)[1]
    f_table = output_file.create_earray(node, 'mfcc', f_atom, (0, feature_shape),
                                        expectedrows = 150)
    
    # append new data to the tables
    f_table.append(features)
    output_file.create_array(node, 'alignment', ali)
    output_file.create_array(node, 'transcript', bytes(transcript, 'utf-8'))
    output_file.create_array(node, 'lemma', bytes(lemma, 'utf-8'))
    output_file.create_array(node, 'word_type', bytes(word_type, 'utf-8'))
    output_file.create_array(node, 'speaker_id', bytes(s_id, 'utf-8'))
    output_file.create_array(node, 'word_form', bytes(word_form, 'utf-8'))