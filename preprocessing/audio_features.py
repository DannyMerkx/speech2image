#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:41:59 2016
Main script for generating audio features such as filterbanks and mfccs
may 2020: switched to python_speech_features for feature calculation. 
@author: danny
"""
import os
import wave
import tables
import librosa

import numpy as np
import python_speech_features.base as base
import python_speech_features.sigproc as sigproc

def fix_wav(path_to_file):
    # In the flickr dataset there is one wav file with an incorrect header, 
    # causing it to be unreadable by wav read. This opens the file with the 
    # wave package, extracts the correct number of frames and saves the file 
    # with a correct header

    file = wave.open(path_to_file, 'r')
    # derive the correct number of frames from the file
    frames = file.readframes(file.getnframes())
    # get all other header parameters
    p = file.getparams()
    file.close()
    # save the file with a new header containing the correct number of frames
    out_file = wave.open(path_to_file, 'w')
    out_file.setparams(out_file.getparams())
    out_file.writeframes(frames)
    out_file.close()

# extract the audio features, params contains most of the settings for feature 
# extraction, img_audio is a dictionary mapping each img to the corresponding
# audio files, append_name is some arbitrary name which has to start with a 
# letter (required by pytables).
def audio_features (params, img_audio, audio_path, append_name, node_list):    
    output_file = params['output_file']
    # create pytable atom for the features   
    f_atom= tables.Float32Atom() 
    count = 1
    # keep track of the nodes for which no features could be made, places 
    # database contains some empty audio files
    invalid = []
    for node in node_list:
        print(f'processing file: {count}')
        count+=1
        # create a group for the desired feature type
        audio_node = output_file.create_group(node, params['feat'])
        # get the base name of the node this feature will be appended to
        base_name = node._v_name.split(append_name)[1]
        # get the caption file names corresponding to the image of this node
        caption_files = img_audio[base_name][1]
        
        for cap in caption_files:
            # remove extension from the caption filename
            base_capt = cap.split('.')[0]
            # remove folder path from file names (Places/coco database)
            if '/' in base_capt:
                base_capt = base_capt.split('/')[-1]
            if '-' in base_capt:
                base_capt = base_capt.replace('-', '_')
            # read audio samples
            try:
                input_data, fs = librosa.load(os.path.join(audio_path, cap),
                                              sr = None)
                # in the places database some of the audiofiles are empty
                if len(input_data) == 0:    
                    break
            except:
                # try to repair broken files, some files had a wrong header. 
                # In Places I found some that could not be fixed however
                try:
                    fix_wav(os.path.join(audio_path, cap))
                    #input_data = read(os.path.join(audio_path, cap))
                except:
                    # the loop will break, if no valid audio features could 
                    # be made for this image, the entire node is deleted.
                    break           
            # set the fft size to the power of two equal to or greater than 
            # the window size.
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
            
            # apply cepstral mean variance normalisation
            if params['normalise']:
                features = (features - features.mean(0))/features.std(0)
            # optionally add the deltas and double deltas
            if params['use_deltas']:
                
                single_delta = base.delta(features, params['delta_n'])
                double_delta = base.delta(single_delta, params['delta_n'])
                features= np.concatenate([features, single_delta, 
                                          double_delta], 1
                                         )
###############################################################################               
            # create new leaf node in the feature node for the current audio 
            # file
            feature_shape= np.shape(features)[1]
            f_table = output_file.create_earray(audio_node, 
                                                append_name + base_capt, 
                                                f_atom, (0, feature_shape),
                                                expectedrows = 5000
                                                )
        
            # append new data to the tables
            f_table.append(features)
        if audio_node._f_list_nodes() == []:
            # keep track of all the invalid nodes for which no features could 
            # be made
            invalid.append(node._v_name)
            # remove the top node including all other features if no captions 
            # features could be created
            output_file.remove_node(node, recursive = True)
    print(invalid)
    print(f'There were {len(invalid)} files that could not be processed')