#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 15:58:13 2018

@author: danny
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:04:54 2018

@author: danny
extract vgg16 features using pretrained network and add them to an h5 file. 
to-do: vgg16 weight location is hard coded, add option to pass it through prep_features.py

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

# this script uses a pretrained vgg16 model to extract the penultimate layer activations
# for images

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


def vgg(img_path, output_file, append_name, img_audio, node_list):
    # initialise the pretrained model
    model = VGG_16('/data/vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    # function to get penultimate layer activations
    get_activations = theano.function([model.layers[0].input, K.learning_phase()], model.layers[34].output)
    # atom defining the type of the image features that will be appended to the output file    
    img_atom= tables.Float32Atom()
    
    count = 1
    # loop through all nodes (the main function creates a h5 file with an empty node for each image file)
    for node in node_list:
        print('processing file:' + str(count))
        count+=1
        # split the appended name from the node name to get the dictionary key for the current file
        base_name = node._v_name.split(append_name)[1]
        # strip the appended naming convention from the group name to be able to retrieve the file
        img_file = img_audio[base_name][0]
        # name for the img node is the same as img_file name except for the places database were the relative path is included 
        node_name = img_file.split('.')[0]
        if '/' in node_name:
                node_name = node_name.split('/')[-1]
        # read and resize the image
        im = image_resize(cv2.imread(os.path.join(img_path, img_file)).astype(np.float32), 256)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        crops = crop_10(im, 224)
        crops = crops.transpose((0,3,1,2))
        im = np.expand_dims(im, axis=0)
        # get the activations of the penultimate layer, 0 indicates the network is used in test mode (no dropout applied)
        activations = (get_activations(im, 0)) 
        # get the shape of the image features for the output file
        feature_shape= numpy.shape(activations)[1]
        vgg_node = output_file.create_group(node, 'vgg')
        # create a pytable array named 'vgg' at the current image node. Remove file extension from filename as dots arent allowed in pytable names
        vgg_array = output_file.create_earray(vgg_node, append_name + node_name, img_atom, (0,feature_shape), expectedrows=1)
        # append the vgg features to the array
        vgg_array.append(activations)
        
def image_resize(image, size, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    if h <= w:
        height = size
    else:
        width = size
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def crop_10 (im, size):
    crops = []
    
    crops.append(im[:224,:224,:])
    crops.append(im[:224,-224:,:])
    crops.append(im[-224:,:224,:])
    crops.append(im[-224:,-224:,:])
    (h,w) = im.shape[:2]
    h_margin = int((h - size)/2)
    w_margin = int((w - size)/2)
    crops.append(im [h_margin:h_margin + size, w_margin: w_margin +size])
    
    im = cv2.flip(im, 1)
    crops.append(im[:224,:224,:])
    crops.append(im[:224,-224:,:])
    crops.append(im[-224:,:224,:])
    crops.append(im[-224:,-224:,:])
    (h,w) = im.shape[:2]
    h_margin = int((h - size)/2)
    w_margin = int((w - size)/2)
    crops.append(im [h_margin:h_margin + size, w_margin: w_margin +size])
    
    return np.array(crops)