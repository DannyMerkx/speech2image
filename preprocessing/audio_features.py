#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:41:59 2016

@author: danny
"""
from aud_feat_functions import get_fbanks, get_freqspectrum, get_mfcc, delta, raw_frames
from scipy.io.wavfile import read
import numpy
import tables
import os
import wave
#  script for creating the features derived from the spoken captions. 
# fix_wav is included because in the original Flickr database there is a broken
# wav file with a wrong header which causes the script to crash.



def fix_wav(path_to_file):
    #fix wav file. In the flickr dataset there is one wav file with an incorrect 
    #number of frames indicated in the header, causing it to be unreadable by pythons
    #wav read function. This opens the file with the wave package, extracts the correct
    #number of frames and saves the file with a correct header

    file = wave.open(path_to_file, 'r')
    # derive the correct number of frames from the file
    frames = file.readframes(file.getnframes())
    # get all other header parameters
    params = file.getparams()
    file.close()
    # now save the file with a new header containing the correct number of frames
    out_file = wave.open(path_to_file, 'w')
    out_file.setparams(file.getparams())
    out_file.writeframes(x)
    out_file.close()


# extract the audio features, params is a big list containg most of the settings
# for feature extraction, img_audio is a dictionary mapping each img to its corresponding
# audio files, append_name is some arbitrary name which has to start with a letter. The Flickr
# audio file names start with numbers but pytables naming conventions require group names to 
# start with a letter in order to call the nodes and their contents.

def audio_features (params, img_audio, audio_path, append_name, node_list):
    
    output_file = params[5]
    # create pytable atom for the features   
    f_atom= tables.Float32Atom() 
    count = 1
    for node in node_list:
        print('processing file:' + str(count))
        count+=1
        # create a group for the desired feature type (e.g. a group called 'fbanks')
        audio_node = output_file.create_group(node, params[4])
        # get the base name of the node this feature will be appended to
        base_name = node._v_name.split(append_name)[1]
        # get the caption file names corresponding to the image of this node
        caption_files = img_audio[base_name][1]
        
        for cap in caption_files:
            # basename for the caption file, i.e. cut of the file extension as dots arent
	        # allowed in pytables group names. 
            base_capt = cap.split('.')[0]
            # as the places database splits the audio files over multiple subfolders these paths from
            # the top folder are included the captions in the dictionary but can be removed from the base_name
            # of the node in the h5 file. 
            if '/' in base_capt:
                base_capt = base_capt.split('/')[:]
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
        
            # create features (implemented are raw audio, the frequency spectrum, fbanks and
            # mfcc's)
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
            
            # optionally add the frame energy
            if params[7]:
                features = numpy.concatenate([energy[:,None], features],1)
            # optionally add the deltas and double deltas
            if params[6]:
                single_delta= delta (features,2)
                double_delta= delta(single_delta,2)
                features= numpy.concatenate([features,single_delta,double_delta],1)
           
            # create new leaf node in the feature node for the current audio file
            feature_shape= numpy.shape(features)[1]
            f_table = output_file.create_earray(audio_node, append_name + base_capt, f_atom, (0,feature_shape),expectedrows=5000)
        
            # append new data to the tables
            f_table.append(features)

    return 
