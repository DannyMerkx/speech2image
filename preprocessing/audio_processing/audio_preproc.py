#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:47:50 2017
provides some basic preprocessing functions for audio files, such as
padding the frames, hammingwindow for the  frames, data preemphasis and fourrier
transform 
@author: danny
"""
from scipy.fftpack import fft
from scipy.signal import iirnotch, lfilter
import numpy



def four(frames, fs, windowsize):
   # fft works on frames of size 2^x, first find the appropriate padsize for 
   # our framesize.
   exp = 1
   while True:
       if numpy.power(2, exp) - windowsize >= 0:
           padsize= numpy.power(2,exp) - windowsize
           break
       else:
           exp += 1
   # pad frames to be of size 2^x        
   frames = numpy.pad(frames, [(0,0), (0, padsize)], 'constant', 
                      constant_values = 0)
   # set cutoff at the half the frame size (+1 to keep the bin around 
   # which the spectrum is mirrored)
   cutoff = int((windowsize+padsize)/2)+1
   # perform fast fourier transform
   Y = fft(frames)    
   # take absolute power and collapse spectrum. Normalise the power for the
   # amount of bins but multiply by 2 to make up for the collapse of the spectrum
   Yamp = 2/(windowsize+padsize)* numpy.abs(Y[:, 0:cutoff])
   # first amp (dc component) and nyquist freq bin are not to be doubled (as they
   # are not mirrored in the fft)
   Yamp[:,0] = Yamp[:,0]/2
   Yamp[:,-1] = Yamp[:,-1]/2
   return (Yamp)

def notch(data):
# apply a notch filter to remove the DC offset
    b, a = iirnotch(0.001, 3.5)
    notched = lfilter(b, a, data)
    return notched
    
def pad (data, window_size, frame_shift):
    # function to pad the audio file to fit the frameshift
    pad_size = window_size - numpy.mod(data.size, frame_shift) 
    # if needed add padding to the end of the data
    if pad_size > 0:
        data = numpy.append(data, numpy.zeros(int(pad_size)))
    return(data)
  
def preemph(data, alpha):
    # preemphasises the data: x(preemph) = X(t) - X(t-1)*alpha
    xt = data
    xtminus1 = data*alpha
    xtminus1 = numpy.insert(xtminus1,0,0,1)[:,:-1]
    data_preemph = xt-xtminus1  
    return data_preemph
    
def hamming(data):
    # apply hamming windowing to a frame of data
    L = numpy.shape(data)[1]
    hammingwindow = 0.54-(0.46*numpy.cos(2*numpy.pi*(numpy.arange(L)/(L-1))))
    data = numpy.multiply(data, hammingwindow)
    return data
