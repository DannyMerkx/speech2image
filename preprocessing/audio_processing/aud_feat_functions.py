#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:41:47 2017

@author: danny
"""
from audio_preproc import four, pad, preemph, hamming, notch
from filters import apply_filterbanks,filter_centers, create_filterbanks
from scipy.fftpack import dct
import numpy as np
import math

# this file contains the main bulk of the actuall feature creation functions 

def delta (data, N):
# calculate delta features, n is the number of frames to look forward and backward
    
    # create a delta array of the right shape
    dt = np.zeros(data.shape)
    # pad data with first and last frame for size of n
    for n in range (N):
        data = np.row_stack((data[0,:],data, data[-1,:]))            
    # calc n*c[x+n] + c[x-n] for n in Nand sum them
    for n in range (1, N + 1):
       dt += np.array([n * (data[x+n,:] - data[x-n,:]) for x  in range (N, len(data) - N)])
    # normalise the deltas for the size of N
    normalise = 2* sum([np.power(x,2) for x in range (1, N+1)]) 
    
    dt = dt/normalise
        
    return (dt)

def raw_frames(input_data, frame_shift, window_size, apply_notch = False):
# this function cuts the data into frames and calculates each frames' energy

    #determine the number of frames to be extracted
    nframes = math.floor(input_data[1].size/frame_shift)
    # apply notch filter
    if apply_notch:
        input_data[1] = notch(input_data[1])
    # pad the data
    data = pad(input_data[1], window_size, frame_shift)
    # slice the frames from the wav file
    # keep a list with the frames and all the values of the samples and 
    # list with the start and end sample# of each frame
    frames = []

    for f in range (0, nframes):
        frame = data[f * frame_shift : f * frame_shift + window_size]
        frames.append(frame)
    
    frames = np.array(frames)
    return frames

def get_freqspectrum(frames, alpha, fs, window_size):
# this function prepares the raw frames for conversion to frequency spectrum
# and applies fft

    # apply preemphasis
    frames = preemph(frames, alpha)
    # apply hamming windowing
    frames = hamming(frames)
    # apply fft 
    freq_spectrum = four(frames, fs, window_size)
    energy = np.sum(freq_spectrum, 1)
    energy = np.log(np.where(energy == 0, np.finfo(float).eps, energy))
    return freq_spectrum, np.array(energy)

def get_fbanks(freq_spectrum, nfilters, fs):
#  this function calculates the filters and creates filterbank features from
#  the fft features
    
    # get the frequencies corresponding to the bins returned by the fft
    xf = np.linspace(0.0, fs/2, np.shape(freq_spectrum)[1])
    # get the filter frequencies
    fc = filter_centers(nfilters, fs, xf)
    # create filterbanks
    filterbanks = create_filterbanks(nfilters, xf, fc)
    # apply filterbanks
    fbanks = apply_filterbanks(freq_spectrum, filterbanks)    
    
    return fbanks

def get_mfcc(fbanks):
# this function creates mfccs from the fbank features
    
    # apply discrete cosine transform to get mfccs. According to convention, 
    # we discard the first filterbank (which is roughly equal to the method 
    # where we only space filters from 1000hz onwards)
    mfcc = dct(fbanks[:,0:], norm = 'ortho')
    # discard the first coefficient of the mffc as well and take the next 12
    # coefficients.
    mfcc = mfcc[:,1:13]
    
    return mfcc
