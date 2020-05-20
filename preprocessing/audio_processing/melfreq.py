#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:54:50 2017

@author: danny
"""
import numpy
#provides simple functions to convert a frequency to mel and vice versa

def freq2mel(f):
    #converts a frequency to mel
    mel=1125*numpy.log(1+f/700)
    return (mel)

def mel2freq(m):
    #converts mel to frequency
    f=700*(numpy.exp(m/1125)-1)
    return (f)