#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:13:04 2021

@author: danny
"""
import numpy as np
import torch
import tables

import sys
sys.path.append('/home/danny/Documents/project_code/speech2image/PyTorch/functions')
sys.path.append('/home/danny/Documents/project_code/speech2image/PyTorch/flickr_audio')
from encoder_configs import create_encoders

# functions to embed the images and utterances
def embed_images(img_embedder, images):
    img_embedder.eval()
    image = dtype()
    for im in images:
        im = im.resnet.read()
        im = dtype(im)
        im.requires_grad = False
        image = torch.cat((image, im.unsqueeze(0).data))
    image = img_embedder(image)
    return image.data

def embed_utt(audio_embedder, utt):
    utt = utt.mfcc_81.read()
    l = [utt.shape[0]]
    utt = dtype(utt).t()
    utt.requires_grad = False
    return audio_embedder(utt.unsqueeze(0), l).data

def cosine(emb_1, emb_2):
    # cosine expects embeddings to be normalised in the embedder
    return torch.matmul(emb_1, emb_2.t())

data_loc = './test_features.h5'
cap_model = '../models/caption_model.24'
img_model = '../models/image_model.24'
img_net, cap_net = create_encoders('rnn')

if torch.cuda.is_available():
    cap_state = torch.load(cap_model)
    img_state = torch.load(img_model)
else:
    cap_state = torch.load(cap_model, map_location = torch.device('cpu'))
    img_state = torch.load(img_model, map_location = torch.device('cpu'))

cap_net.load_state_dict(cap_state)
img_net.load_state_dict(img_state)

cap_net.eval()
img_net.eval()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

data = tables.open_file(data_loc, mode='a')
# load the data, split the utterances into verbs and nouns
images = data.root.image._f_list_nodes()
utterances = data.root.audio._f_list_nodes()

nouns = []
verbs = []
for utt in utterances:
    if utt.word_type.read().decode() == 'noun':
        nouns.append(utt)
    elif utt.word_type.read().decode() == 'verb':
        verbs.append(utt)

encoded_images = embed_images(img_net, images)

utt = nouns[0]
utt_emb = embed_utt(cap_net, utt)

sim = cosine(utt_emb, encoded_images)

top_ims = sim.topk(10)

def precision_at_n(utterances, encoded_images, images):
    precision = []
    for utt in utterances:
        utt_emb = embed_utt(cap_net, utt)
        sim = cosine(utt_emb, encoded_images)
        top_ims = sim.topk(10)
    
        p_at_n = 0
        for idx in top_ims[1][0]:
            image = images[idx]
            word_type = utt.word_type.read().decode()
            if word_type == 'noun':
                annot = image.nouns.read()
            elif word_type == 'verb':
                annot = image.verbs.read()
                
            lemma = utt.lemma.read()
            if lemma in annot:
                p_at_n += 1
        
        if utt.word_form.read().decode() == 'singular':
            print(f'{utt.transcript.read().decode()}: {p_at_n}')
            precision.append(p_at_n)
    return precision

p = precision_at_n(nouns, encoded_images, images)
    














