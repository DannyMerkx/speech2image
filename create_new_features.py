#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:10:41 2018

@author: danny
"""
import numpy as np
import tables
# iterator which return the sum of some feature and the lenght of the feature,
# e.g. to calculate speech feature mean for normalisation
def feature_sum(f_nodes):
    for f in f_nodes:
        features = f[:]
        f.close()
        yield np.sum(features,0), features.shape[0]
# same but for variance normalisation
def feature_var(f_nodes, f_mean):
    for f in f_nodes:
        features = np.power(f[:] - f_mean, 2)
        f.close()
        yield np.sum(features,0), features.shape[0]

# iterator which returns all nodes of a given feature
def iterate_nodes(h5_file, feature_name):
    for x in h5_file.root:
        for y in x:
            if y._v_name == feature_name:
                for z in y:
                    yield z

# create a new feature from an existing feature. takes an opened h5_file,
# name of the original feature and new feature, pytables atom for the new feature,
# a function to calculate new features and its arguments (use list format for multiple arguments)
def create_feature(h5_file, or_feature_name, new_feature_name, f_atom, function, arguments):
    for node in h5_file.root:
        or_features = [feats for feats in eval('node.' + or_feature_name)]
        new_feature_node = h5_file.create_group(node, new_feature_name)
        for feat in or_features:
            name = feat._v_name
            new_feat = function(feat, arguments)
            feature_shape= np.shape(new_feat)[1]
            feature_table = h5_file.create_earray(new_feature_node, name, f_atom,(0,feature_shape), expectedrows= 5000)
            feature_table.append(new_feat)
# Functions to pass to the create_feature method
###############################################################################

# mean normalisation, takes a feature node and a database mean vector
def mean_normalise(features, database_mean):
    # pad or truncate the speech signal
    while np.shape(features)[0]<1024:
                features = np.concatenate((features, np.zeros([1,40])),0)
    if np.shape(features)[0] >1024:
                features = features[:1024,:]
    return features[:] - database_mean
# mean and variance normalisation, args is a list with a 
# mean int or array first and a variance int or array second
def mean_var_normalise(features, args):
    # pad or truncate the speech signal
    while np.shape(features)[0]<1024:
                features = np.concatenate((features, np.zeros([1,40])),0)
    if np.shape(features)[0] >1024:
                features = features[:1024,:]
    return (features[:] - args[0]) / np.sqrt(args[1])

###############################################################################
    
output_file = tables.open_file('/home/danny/Documents/Flickr/flickr_features.h5', mode='a')
f_atom= tables.Float32Atom()

# calculate the mean and variance
fbanks = [x for x in iterate_nodes(output_file, 'fbanks')]
f_mean = np.zeros(fbanks[0].shape[1])
n_frames = 0              
for f in feature_sum(fbanks):
    n_frames += f[1]
    f_mean += f[0] 
# take the mean and variance per filterbank or over all filterbanks     
f_mean = f_mean / n_frames
# f_mean = np.mean(f_mean)/ n_frames

fbanks = [x for x in iterate_nodes(output_file, 'fbanks')]
f_var = np.zeros(fbanks[0].shape[1])
n_frames = 0              
for f in feature_var(fbanks, f_mean):
    n_frames += f[1]
    f_var += f[0]   

f_var = f_var / n_frames
# f_var = np.mean(f_var) / n_frames

# this creates the fbanks in the same manner as in the Harwath and Glass paper
# i.e. mean normalised with one mean for the entire database.
create_feature(output_file, 'fbanks', 'HG_fbanks', f_atom, mean_normalise, f_mean)

#  mean and variance normalised fbanks
create_feature(output_file, 'fbanks', 'meanvar_norm_fbanks', f_atom, mean_var_normalise, [f_mean, f_var])

output_file.close()