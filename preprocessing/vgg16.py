#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:04:54 2018

@author: danny
"""
import theano
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import os
import numpy
import tables

from keras import backend as K
K.set_image_dim_ordering('th')

# pretrained vgg_16 model
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def vgg(img_path, output_file, append_name):
    # initialise the pretrained model
    model = VGG_16('/home/danny/Documents/Flickr/vgg16_weights.h5')
    #model = VGG_16(os.path.join('C:\\','Users', 'Beheerder','Documents','PhD','Flickr','vgg16_weights.h5'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    # function to get penultimate layer activations
    get_activations = theano.function([model.layers[0].input, K.learning_phase()], model.layers[34].output)
    # atom defining the type of the image features that will be appended to the output file    
    img_atom= tables.Float32Atom()
    
    count = 1
    # loop through all nodes (the main function creates a h5 file with an empty node for each image file)
    for node in output_file.root:
        print('processing file:' + str(count))
        count+=1
        base_name = node._v_name.split(append_name)[1]
        # strip the appended naming convention from the group name to be able to retrieve the file
        filename = base_name + '.jpg'
        # read and resize the image
        im = cv2.resize(cv2.imread(os.path.join(img_path, filename)), (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        # get the activations of the penultimate layer, 0 indicates the network is used in test mode (no dropout applied)
        activations = (get_activations(im, 0)) 
        # get the shape of the image features for the output file
        feature_shape= numpy.shape(activations)[1]
        vgg_node = output_file.create_group(node, 'vgg')
        # create a pytable array named 'vgg' at the current image node. Remove file extension from filename as dots arent allowed in pytable names
        vgg_array = output_file.create_earray(vgg_node, append_name + base_name, img_atom, (0,feature_shape), expectedrows=1)
        # append the vgg features to the array
        vgg_array.append(activations)
