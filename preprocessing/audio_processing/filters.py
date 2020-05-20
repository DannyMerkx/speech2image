#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:23:01 2017

@author: danny
"""
# functions for creating and applying the filter banks
from melfreq import freq2mel, mel2freq
import numpy

def create_filterbanks (nfilters, freqrange, fc):
    # function to create filter banks. takes as input
    # the number of filters to be created, the frequency range and the
    # filter centers
    filterbank = []
    # for the desired # of filters do
    for n in range (0,nfilters):
        # set the begin center and end frequency of the filters
        begin = fc[n]
        center= fc[n+1]
        end = fc[n+2]
        f = []
        # create triangular filters
        for x in freqrange:
            # 0 for f outside the filter
            if x < begin:
                f.append(0)
            #increasing to 1 towards the center
            elif begin <= x and x < center:
                f.append((x-begin)/(center-begin))
            elif x == center:
                f.append(1)
            # decreasing to 0 upwards from the center
            elif center < x and x <= end:
                f.append((end-x)/(end-center))
            # 0 outside filter range
            elif x > end:
                f.append(0)
                
        filterbank.append(f)
        
    return filterbank
    
def filter_centers(nfilters, fs, xf):
    # calculates the center frequencies for the mel filters
    
    # space the filters equally in mels
    spacing = numpy.linspace(0, freq2mel(fs/2), nfilters+2)
    #back from mels to frequency
    spacing = mel2freq(spacing)
    # round the filter frequencies to the nearest availlable fft bin frequencies
    # and return the centers for the filters.  
    filters = [xf[numpy.argmin(numpy.abs(xf-x))] for x in spacing]    
    
    return filters
    
def apply_filterbanks(data, filters):
    # function to apply the filterbanks and take the log of the filterbanks
    filtered_freq = numpy.dot(data, numpy.transpose(filters))
    filtered_freq = numpy.log(numpy.where(filtered_freq == 0, 
                                          numpy.finfo(float).eps, 
                                          filtered_freq
                                          )
                              )
    return filtered_freq