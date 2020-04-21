#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:04:54 2018

@author: danny
extract visual features for images using pretrained networks

"""
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import PIL.Image
import tables

# this script uses a pretrained model to extract the penultimate layer 
# activations for images

# creates features from the vgg19 model minus the penultimate classificaton layer
def vgg19():
    # initialise a pretrained model, see torch docs for the availlable 
    # pretrained models. Torch will download the weights automatically
    model = models.vgg19_bn(pretrained = True)

    # remove the final layer from the classifier. 
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier
    return model

# resnet model minus the penultimate classificaton layer
def resnet():
    model = models.resnet152(pretrained = True)
    model = nn.Sequential(*list(model.children())[:-1])
    return model

# creates features from the resnet model minus final three layers
def resnet_truncated():
    model = models.resnet152(pretrained = True)
    model = nn.Sequential(*list(model.children())[:-3])
    return model

# resize and take ten crops of the image. Return the average activations over 
# the crops
def prep_tencrop(im, model):
    for p in model.parameters():
    	p.requires_grad = False
    model.eval()
    
    # define the required functions. Normalisation parameters are hardcoded
    # (average over the network's training data)
    tencrop = transforms.TenCrop(224)
    tens = transforms.ToTensor()
    normalise = transforms.Normalize(mean = [0.485,0.456,0.406], 
                                     std = [0.229, 0.224, 0.225])
    resize = transforms.Resize(256, PIL.Image.ANTIALIAS)
    
    im = tencrop(resize(im))
    im = torch.cat([normalise(tens(x)).unsqueeze(0) for x in im])
    if torch.cuda.is_available():
        im = im.cuda()
    # expand greyscale images to 3 channels so the visual networks accept them
    if not im.size()[1] == 3:
        im = im.expand(im.size()[0], 3, im.size()[2], im.size()[3])
    activations = model(im)
    return activations.mean(0).squeeze()

# only resize the image and get the model activations for a single image
def prep_resize(im, model):
    for p in model.parameters():
    	p.requires_grad = False
    model.eval()
        
    tens = transforms.ToTensor()
    normalise = transforms.Normalize(mean = [0.485,0.456,0.406], 
                                     std = [0.229, 0.224, 0.225])
    resize = transforms.Resize((224,224), PIL.Image.ANTIALIAS)
    
    im = resize(im)
    im = normalise(tens(im))
    if torch.cuda.is_available():
        im = im.cuda()

    if not im.size()[0] == 3:
        im = im.expand(3, im.size()[2], im.size()[3])
    activations = model(im.unsqueeze(0))
    return activations.squeeze()

def prep_raw(im, model):
    tencrop = transforms.TenCrop(224)
    tens = transforms.ToTensor()
    normalise = transforms.Normalize(mean = [0.485,0.456,0.406], 
                                     std = [0.229, 0.224, 0.225])
    resize = transforms.Resize(256, PIL.Image.ANTIALIAS)
    
    im = tencrop(resize(im))
    im = torch.cat([normalise(tens(x)).unsqueeze(0) for x in im])
    if torch.cuda.is_available():
        im = im.cuda()
    if not im.size()[1] == 3:
        im = im.expand(im.size()[0], 3, im.size()[2], im.size()[3])
    return im

def vis_feats(img_path, output_file, append_name, img_audio, node_list, net):
    # prepare the pretrained model
    if net == 'vgg19':
        model = vgg19()
        get_activations = prep_tencrop
    if net == 'resnet':
        model = resnet()
        get_activations = prep_tencrop
    if net == 'resnet_trunc':
        model = resnet_truncated()
        get_activations = prep_resize
    if net == 'raw':
        get_activations = prep_raw
        model = []
    # set the model to use cuda
    if torch.cuda.is_available() and model:
        model = model.cuda()
    
    count = 1
    # loop through all nodes (the main function creates a h5 file with an 
    # empty node for each image file)
    for node in node_list:
        print('processing file:' + str(count))
        count+=1
        # split the appended name from the node name to get the dictionary key
        # for the current file
        base_name = node._v_name.split(append_name)[1]
        # strip the appended naming convention from the group name to be able 
        # to retrieve the file
        img_file = img_audio[base_name][0]
        # name for the img node is the same as img_file name except for the 
        # places database were the relative path is included 
        node_name = img_file.split('.')[0]
        if '/' in node_name:
                node_name = node_name.split('/')[-1]
        
        im = PIL.Image.open(os.path.join(img_path, img_file))
        activations = get_activations(im, model)

        # create a new node 
        vis_node = output_file.create_group(node, net)
        # create a pytable array at the current image node. Remove file 
        # extension from filename as dots arent allowed in pytable names
        vis_array = output_file.create_array(vis_node, append_name + node_name, 
                                             activations.data.cpu().numpy())
