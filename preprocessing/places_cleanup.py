#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:28:36 2018
Cleanup the places images database by removing all the images for which no 
captions exist. Run once before creating features for the places database
@author: danny
"""
import os
import json
import glob

# location of the json files
meta_data_loc = '/vol/tensusers3/dmerkx/databases/places/'
# location of the jpeg files
img_path = os.path.join('/vol/tensusers3/dmerkx/databases/places/data/')
# load metadata for the train and val sets
val = json.load(open(os.path.join(meta_data_loc, 'val.json')))
train = json.load(open(os.path.join(meta_data_loc, 'train.json')))

imgs = val['data'] + train['data']

imgs= [os.path.join(img_path, x['image']) for x in imgs]

x = glob.glob(img_path + '/*/*.jpg')
y = glob.glob(img_path + '/*/*/*.jpg')
places_data = x + y 

# check if all files in the metadata are actually in the places database
union = set(places_data) & set(imgs)
print(len(union) == len(imgs))

disjunct = set(places_data) - set(imgs)

# remove places images for which there is no caption to save disk space
for x in list(disjunct):
    os.remove(x)
