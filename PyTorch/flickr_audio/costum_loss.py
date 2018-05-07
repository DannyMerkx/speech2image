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

# hinge loss function based on a symmetric distance measure. The loss function uses the dot product,
# you get the cosine similarity by normalising your embeddings at the output layer of the encoders.
def batch_hinge_loss(embeddings_1, embeddings_2, cuda = True):
    # batch size
    batch_size = embeddings_1.size(0)   

    # calculate the similarity score
    error = - torch.matmul(embeddings_1, embeddings_2.t())

    # get the similarity of the correct image-caption pairs (the diagonal of the similarity matrix)
    I = torch.autograd.Variable(torch.eye(batch_size), requires_grad = True)
    if cuda:
        I = I.cuda()
    diag = (error * I).sum(dim=0)
    
    # calculate the image to text and text to image cost.
    cost_1 = torch.clamp(.2 - error + diag, min = 0)
    cost_2 = torch.clamp(.2 - error + diag.view(-1, 1), min = 0)
    cost = cost_1 + cost_2
    
    # remove the diagonal for the cost matrix (i.e. count no costs for correct pairs)
    I_2 = torch.autograd.Variable(torch.eye(batch_size), requires_grad = True)
    if cuda:
        I_2 = I_2.cuda()
    
    cost = (1 - I_2) * cost
    return cost.mean()

#implements the ordered embeddings loss function proposed by vendrov et all.
def ordered_loss(embeddings_1, embeddings_2, cuda = True):
    # batch size
    batch_size = embeddings_1.size(0)   
    # calculate the norms of the embeddings and normalise the embeddings
    if norm[0]:
        embeddings_1 = embeddings_1 / embeddings_1.norm(2, dim = 1, keepdim = True)
    if norm[1]:
        embeddings_2 = embeddings_2 / embeddings_2.norm(2, dim = 1, keepdim = True)
    
    # calculate the similarity score as described by vendrov et al. the partial order is image < caption, 
    # i.e. the captions are abstractions of the images (the wrong order results in worse results). 
    err = - (torch.clamp(embeddings_1 - embeddings_2[0], min = 0).norm(1, dim = 1, keepdim = True)**2)
    for x in range(1, embeddings_2.size(0)):
        e =  - (torch.clamp(embeddings_1 - embeddings_2[x], min = 0).norm(1, dim = 1, keepdim = True)**2)
        err = torch.cat((err, e), 1)

    # get the similarity of the correct image-caption pairs    
    I = torch.autograd.Variable(torch.eye(batch_size), requires_grad = True)
    if cuda:
        I = I.cuda()
    diag_1 = (err * I).sum(dim=0)

    # calculate the image to text and text to image cost.
    cost_1 = torch.clamp(.2 - diag_1 + err, min = 0)
    cost_2 = torch.clamp(.2 - diag_1.view(-1, 1) + err, min = 0)    
    cost = cost_1 + cost_2
    
    # remove the diagonal for the cost matrix (i.e. count no costs for correct pairs)
    I_2 = torch.autograd.Variable(torch.eye(batch_size), requires_grad = True)
    if cuda:
        I_2 = I_2.cuda()
    
    cost = (1 - I_2) * cost
    
    return cost.mean()
