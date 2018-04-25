#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:10:41 2018

@author: danny

This script can be used to create new features from an existing feature e.g. by normalising the mel filterbanks. 
The create feature function is a general function that takes another function and some arguments and applies that
function to all of the indicated data and creates a new feature group in the appropriate structure for the network 
training to read.  
"""
import numpy as np
import tables
import argparse

parser = argparse.ArgumentParser(description='derive new features from an existing feature in the datafile e.g. feature normalisation')

parser.add_argument('-data_loc', type = str, default = '/prep_data/flickr_features.h5',
                    help = 'location of the feature file, default: /prep_data/flickr_features.h5')

args = parser.parse_args()

# iterator which returns all nodes of a given feature. Usefull for collecting all nodes 
# of some feature e.g. for calculating the mean or variance of that feature. Takes an opened
# h5 file and the group name of the features

# this iterator is specifically for the file structure used in the flickr_database
def iterate_flickr(h5_file, feature_name):
    for x in h5_file.root:
        for y in x:
            if y._v_name == feature_name:
                for z in y:
                    yield z
# this iterator is specifically for the structure in the places database                    
def iterate_places(h5_file, feature_name):
    for x in h5_file.root:
        for y in x:
            for z in y:                
                if z._v_name == feature_name:
                    for q in z:
                        yield q

# this iterator is more flexible as it doesnt care about the depth of the nodes, however
# recursively listing all nodes and then selecting the nodes you need is slower then a targeted search
def recursive_iter(h5_file, feature_name):
    # iterator which finds all EArrays (the feature nodes)
    node_iter = h5_file.walk_nodes(h5_file.root, 'EArray')
    # select only those features given by feature_name
    for x in node_iter:
        if x._v_parent._v_name == feature_name:
            yield x

iterate_nodes = iterate_flickr
    
################################################################################
##### Functions for calculating arguments to use in new feature creation #######

# it is important to close the feature nodes!!! When inspecting or retrieving
# the contents of a node it is loaded to memory, python does not seem to remove
# it automatically so iterators like below will otherwise load the whole file to 
# memory. (interestingly this does not seem to happen with the mini batch loaders of the NN
# training)

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


##################################################################################
#################### Function for deriving the new features ######################

# create a new feature from an existing feature. Takes an opened h5_file,
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
	    # adjusting the expected rows to be appropriate for your features size saves space
            feature_table = h5_file.create_earray(new_feature_node, name, f_atom,(0,feature_shape), expectedrows= 5000)
            feature_table.append(new_feat)

###############################################################################
###################  Feature functions ########################################

# mean normalisation, takes a feature node and a database mean vector (i.e. per filterbank)
# or scalar (one value over the entire database)
def mean_normalise(features, database_mean):
    # pad or truncate the speech signal
    while np.shape(features)[0]<1024:
                features = np.concatenate((features, np.zeros([1,40])),0)
    if np.shape(features)[0] >1024:
                features = features[:1024,:]
    return features[:] - database_mean

# mean and variance normalisation, args is a list with a 
# mean scalar or array first and a variance scalar or array second (scalar = one
# value for entire database, array = one value per filterbank over entire database)
def mean_var_normalise(features, args):
    # pad or truncate the speech signal
    while np.shape(features)[0]<1024:
                features = np.concatenate((features, np.zeros([1,40])),0)
    if np.shape(features)[0] >1024:
                features = features[:1024,:]
    return (features[:] - args[0]) / np.sqrt(args[1])

###############################################################################

# open the datafile    
output_file = tables.open_file(args.data_loc, mode='a')
# pytables atom for the new feature
f_atom = tables.Float32Atom()

# calculate the mean
fbanks = [x for x in iterate_nodes(output_file, 'fbanks')]
f_mean = np.zeros(fbanks[0].shape[1])
n_frames = 0              
for f in feature_sum(fbanks):
    n_frames += f[1]
    f_mean += f[0] 

f_mean = np.mean(f_mean)/ n_frames

# calculate the variance
fbanks = [x for x in iterate_nodes(output_file, 'fbanks')]
f_var = np.zeros(fbanks[0].shape[1])
n_frames = 0              
for f in feature_var(fbanks, f_mean):
    n_frames += f[1]
    f_var += f[0]   

f_var = np.mean(f_var) / n_frames

# this creates the fbanks in the same manner as in the Harwath and Glass paper
# i.e. mean normalised with one mean for the entire database.
#create_feature(output_file, 'fbanks', 'HG_fbanks', f_atom, mean_normalise, f_mean)

#  mean and variance normalised fbanks
create_feature(output_file, 'fbanks', 'hw_norm_fbanks', f_atom, mean_var_normalise, [f_mean, f_var])

output_file.close()
