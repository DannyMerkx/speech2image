#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:41:59 2016

@author: danny
"""
from label_func import label_frames, parse_transcript
from data_functions import list_files, check_files
from create_features import get_fbanks, get_freqspectrum, get_mfcc, delta, raw_frames
from scipy.io.wavfile import read
import numpy
import tables
import os

# the processing pipeline can be run in two ways. Either just create features (raw frames
# frequency spectrum, filterbanks or mfcc) or create both features and label them.

def features_and_labels (f_ex, params, l_path, a_path):
    # get list of audio and transcript files 
    audio_files = [a_path+"/"+x for x in list_files(a_path)]
    audio_files.sort()  
    
    label_files = [l_path+"/"+ x for x in list_files (l_path)]
    label_files.sort()
    
    # create h5 file for the processed data
    output_file = tables.open_file(params[5], mode='a')
    
    # create pytable atoms   
    f_atom= tables.Float64Atom()
    # N.B. label size is hard coded. It provides phoneme and 7 articulatory feature
    # labels
    l_atom = tables.StringAtom(itemsize = 5)
    # create a feature and label group branching of the root node
    f_node = output_file.create_group("/", 'features')    
    l_node = output_file.create_group("/", 'labels')     
    
    # check if the audio and transcript files match 
    if check_files(audio_files, label_files, f_ex):
    
        for x in range (0,len(audio_files)):
            print ('processing file ' + str(x) )
            # read audio samples
            input_data = read(audio_files[x])
            # get the basename of the audiofile
            base_name = os.path.basename(audio_files[x][:-4])
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
            f_table = output_file.create_earray(f_node, audio_files[x][-12:-4], f_atom, (0,feature_shape),expectedrows=100000)
            
            # create labels
            
            # read the transcript
            trans = parse_transcript(label_files[x], fs)           
            nframes = features.shape[0]        
            # label frames using the labelled transcript
            labels = numpy.array(label_frames(nframes, trans, frame_shift))
            
            # create new leaf node in the label node
            l_table = output_file.create_earray(l_node, audio_files[x][-12:-4], l_atom, (0,8),expectedrows=100000)
            
            # append new data to the tables
            f_table.append(features)
            l_table.append(labels)
    else:
        # error message in case the audio and transcript lists do not match
        print ('audio and transcript files do not match')
    # close the output files
    output_file.close()
    return 

def features (params,a_path):
    # get list of audio and transcript files 
    audio_files = [a_path+"/"+x for x in list_files(a_path)]
    audio_files.sort()  
    # create h5 file for the processed data
    output_file = tables.open_file(params[5], mode='a')
    
    # create pytable atoms   
    f_atom= tables.Float64Atom() 
    # create a feature and label group branching of the root node
    f_node = output_file.create_group("/", 'features')        

    for x in range (0,len(audio_files)):
        print ('processing file ' + str(x) )
        # read audio samples
        input_data = read(audio_files[x])
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
        f_table = output_file.create_earray(f_node, audio_files[x][-12:-4], f_atom, (0,feature_shape),expectedrows=100000)
        
            # append new data to the tables
        f_table.append(features)
    # close the output files
    output_file.close()
    return 
