#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:41:59 2016

@author: danny
"""
from create_features import get_fbanks, get_freqspectrum, get_mfcc, delta, raw_frames
from scipy.io.wavfile import read
import numpy
import tables
import os
import wave
# the processing pipeline can be run in two ways. Either just create features (raw frames
# frequency spectrum, filterbanks or mfcc) or create both features and label them.



def fix_wav(path_to_file):
    #fix wav file. In the flickr dataset there is one wav file with an incorrect 
    #number of frames indicated in the header, causing it to be unreadable by pythons
    #wav read function. This opens the file with the wave package, extracts the correct
    #number of frames and saves the file with a correct header

    file = wave.open(path_to_file, 'r')
    frames = file.readframes(file.getnframes())
    params = file.getparams()
    file.close()

    out_file = wave.open(path_to_file, 'w')
    out_file.setparams(file.getparams())
    out_file.writeframes(x)
    out_file.close()



def audio_features (params, img_audio, audio_path, append_name):
    
    output_file = params[5]
    # create pytable atom for the features   
    f_atom= tables.Float64Atom() 
    count = 1
    for node in output_file.root:
        print('processing file:' + str(count))
        count+=1
        # create a group for the desired feature type
        audio_node = output_file.create_group(node, params[4])
        # base name for the image
        base_name = node._v_name.split(append_name)[1]
        # get the corresponding caption file names
        caption_files = img_audio[base_name + '.jpg']
        
        for cap in caption_files:
            # basename for the caption file. 
            base_capt = cap.split('.')[0]
            # read audio samples
            try:
                input_data = read(os.path.join(audio_path, cap))
            except:
                fix_wav(os.path.join(audio_path, cap))
                input_data = read(os.path.join(audio_path, cap))
                break
            # sampling frequency
            fs = input_data[0]
            # get window and frameshift size in samples
            window_size = int(fs*params[2])
            frame_shift = int(fs*params[3])
        
            # create features
            if params[4] == 'raw':
                [features, energy] = raw_frames(input_data, frame_shift, window_size)
        
            elif params[4] == 'freq_spectrum':
                [frames, energy] = raw_frames(input_data, frame_shift, window_size)
                features = get_freqspectrum(frames, params[0], fs, window_size)
        
            elif params[4] == 'fbanks':
                [frames, energy] = raw_frames(input_data, frame_shift, window_size)
                freq_spectrum = get_freqspectrum(frames, params[0], fs, window_size)
                features = get_fbanks(freq_spectrum, params[1], fs) 
            
            elif params[4] == 'mfcc':
                [frames, energy] = raw_frames(input_data, frame_shift, window_size)
                freq_spectrum = get_freqspectrum(frames, params[0], fs, window_size)
                fbanks = get_fbanks(freq_spectrum, params[1], fs)
                features = get_mfcc(fbanks)
            
            # add the frame energy if needed
            if params[7]:
                features = numpy.concatenate([energy[:,None], features],1)
            # add the deltas and double deltas if needed
            if params[6]:
                single_delta= delta (features,2)
                double_delta= delta(single_delta,2)
                features= numpy.concatenate([features,single_delta,double_delta],1)
           
            # create new leaf node in the feature node
            feature_shape= numpy.shape(features)[1]
            #Remove file extension from filename as dots arent allowed in pytable names
            f_table = output_file.create_earray(audio_node, append_name + base_capt, f_atom, (0,feature_shape),expectedrows=5000)
        
            # append new data to the tables
            f_table.append(features)

    return 
