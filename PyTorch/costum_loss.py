#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:08:28 2018

@author: danny
Costum loss functions for use with theano. Currently includes hinge loss functions using the 
Dot product and cosine similarity as similarity measures for the embeddings

L2norm is somewhat in between cosine and dot product. It normalises the magnitude of 1 of the 2 embeddings, like in the original paper by Harwath and Glass where only the speech embeddings are l2 normalised. According to them this gave better results than the dot product or normalising both embeddings (i.e. cosine similarity)

"""


import torch
import torch.nn as nn

# batch hinge loss function using dot product similarity measure
def dot_hinge_loss(embeddings_1, embeddings_2):
    # batch size
    batch_size = embeddings_1.size(0)
    # calculate the dot product
    dot_prod = torch.mm(embeddings_1, embeddings_2.t())
    
    # get the similarity of the correct image-caption pairs (the diagonal of the similarity matrix)
    matched = dot_prod.diag()
    # get the average mismatch of the image with incorrect captions
    # sum the matrix along the corresponding axis, correct for including the correct pair, divide by batch size -1
    # (also to correct for including the correct pair)
    mismatch_1 = (dot_prod.sum(dim = 0) - matched) / (batch_size - 1)
    # get the average mismatch of the captions with incorrect images
    mismatch_2 = (dot_prod.sum(dim = 1) - matched) / (batch_size - 1)

    return torch.sum(nn.functional.relu(mismatch_1 - matched + 1) + nn.functional.relu(mismatch_2 - matched + 1))

# hinge loss using cosine similarity
def cosine_hinge_loss(embeddings_1, embeddings_2):
    # batch size
    batch_size = embeddings_1.size(0)
    # calculate the numerator
    numerator = torch.mm(embeddings_1, embeddings_2.t())
    # calculate the denominator
    denom1 = torch.sum(torch.pow(embeddings_1, 2), dim = 1)
    denom2 = torch.sum(torch.pow(embeddings_2, 2), dim = 1)
    
    denominator = torch.sqrt(torch.mm(denom1.expand(1,denom1.size(0)).t(), denom2.expand(1,denom2.size(0))))
    # similarity matrix
    sim = numerator/denominator
     # get the similarity of the correct image-caption pairs (the diagonal of the similarity matrix)
    matched = sim.diag()
    # get the average mismatch of the image with incorrect captions
    # sum the matrix along the corresponding axis, correct for including the correct pair, divide by batch size -1
    # (also to correct for including the correct pair)
    mismatch_1 = (sim.sum(dim = 0) - matched) / (batch_size - 1)
    # get the average mismatch of the captions with incorrect images
    mismatch_2 = (sim.sum(dim = 1) - matched) / (batch_size - 1)

    return torch.sum(nn.functional.relu(mismatch_1 - matched + 1) + nn.functional.relu(mismatch_2 - matched + 1))

# hinge loss where only one set of embeddings is l2 normalised like in the original
# harwath and glass paper (l2 normalise of speech embeddings). uncomment the appropriate 
# line to switch which embeddings to normalise, normalising both results in the cosine hinge loss of course. 
def l2norm_hinge_loss(embeddings_1, embeddings_2):
    # batch size
    batch_size = embeddings_1.size(0)   
    # calculate the norms of the embeddings
    
    #denom1 = torch.sqrt(torch.sum(torch.pow(embeddings_1, 2), dim = 1))
    denom2 = torch.sqrt(torch.sum(torch.pow(embeddings_2, 2), dim = 1))   
    
    # calculate the similarity score and normalise one or both embeddings
    sim = torch.mm(embeddings_1, embeddings_2.t()/denom2)
    #sim = torch.mm((embeddings_1.t()/denom1).t(), embeddings_2.t())
    #sim = torch.mm((embeddings_1.t()/denom1).t(), embeddings_2.t()/denom2)
    
    # get the similarity of the correct image-caption pairs (the diagonal of the similarity matrix)
    matched = sim.diag()
    # get the average mismatch of the image with incorrect captions
    # sum the matrix along the corresponding axis, correct for including the correct pair, divide by batch size -1
    # (also to correct for including the correct pair)
    mismatch_1 = (sim.sum(dim = 0) - matched) / (batch_size - 1)
    # get the average mismatch of the captions with incorrect images
    mismatch_2 = (sim.sum(dim = 1) - matched) / (batch_size - 1)

    return torch.sum(nn.functional.relu(mismatch_1 - matched+1) + nn.functional.relu(mismatch_2 - matched+1))
