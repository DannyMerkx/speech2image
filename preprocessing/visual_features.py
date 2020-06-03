#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:04:54 2018

@author: danny
extract visual features for images using pretrained networks
Resnet_truncated can be used to extract resnet activations after N layers, 
which can then be used to finetune the final layers of resnet during training
"""
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import PIL.Image

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
    # takes ten crops of 224x224 pixels
    tencrop = transforms.TenCrop(224)
    tens = transforms.ToTensor()
    # normalises the images using the values provided with torchvision
    normalise = transforms.Normalize(mean = [0.485,0.456,0.406], 
                                     std = [0.229, 0.224, 0.225])
    resize = transforms.Resize(256, PIL.Image.ANTIALIAS)
    
    im = tencrop(resize(im))
    im = torch.cat([normalise(tens(x)).unsqueeze(0) for x in im])
    if torch.cuda.is_available():
        im = im.cuda()
    activations = model(im)
    return activations.mean(0).squeeze()

# only resize the image and get the model activations for a single image
# meant for finetuning pretrained models
def prep_resize(im, model):
    tens = transforms.ToTensor()
    # normalise the images using the values provided with torchvision
    normalise = transforms.Normalize(mean = [0.485,0.456,0.406], 
                                     std = [0.229, 0.224, 0.225])
    # resize the image to 224 x 224
    resize = transforms.Resize((224,224), PIL.Image.ANTIALIAS)
    im = resize(im)
    im = normalise(tens(im))
    if torch.cuda.is_available():
        im = im.cuda()
    activations = model(im.unsqueeze(0))
    return activations.squeeze()
# just normalised and rescaled raw images, for training a costum visual 
# network
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
    # disable gradients and dropout
    for p in model.parameters():
    	p.requires_grad = False
    model.eval()
       
    count = 1
    # loop through all nodes (the main function creates a h5 file with an 
    # empty node for each image file)
    for node in node_list:
        print(f'processing file: {count}')
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
        
        im = PIL.Image.open(os.path.join(img_path, img_file)).convert('RGB')
        activations = get_activations(im, model)

        # create a new node 
        vis_node = output_file.create_group(node, net)
        # create a pytable array at the current image node. Remove file 
        # extension from filename as dots arent allowed in pytable names
        output_file.create_array(vis_node, f'{append_name}{node_name}', 
                                 activations.data.cpu().numpy())
