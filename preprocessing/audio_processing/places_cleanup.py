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
meta_data_loc = '/vol/tensusers3/dmerkx/databases/places/split/'
# location of the jpeg files
img_path = os.path.join('/vol/tensusers3/dmerkx/databases/places/data/')
# load metadata for the train and val sets
places_meta = []
places_meta.append(json.load(open(os.path.join(meta_data_loc, 'dev_seen_2020.json'))))
places_meta.append(json.load(open(os.path.join(meta_data_loc, 'dev_unseen_2020.json'))))
places_meta.append(json.load(open(os.path.join(meta_data_loc, 'test_seen_2020.json'))))
places_meta.append(json.load(open(os.path.join(meta_data_loc, 'test_unseen_2020.json'))))
places_meta.append(json.load(open(os.path.join(meta_data_loc, 'train_2020.json'))))

imgs = [img_path+y['image'] for x in places_meta for y in x['data']]

x = glob.glob(img_path + '/*/*/*.jpg')
y = glob.glob(img_path + '/*/*/*/*.jpg')
places_data = x + y 

# check if all files in the metadata are actually in the places database
union = set(places_data) & set(imgs)
print(len(union) == len(imgs))

disjunct = set(places_data) - set(imgs)

# remove places images for which there is no caption to save disk space
for x in list(disjunct):
    os.remove(x)

    