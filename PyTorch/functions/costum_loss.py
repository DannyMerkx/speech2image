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
def batch_hinge_loss(embeddings_1, embeddings_2, dtype):
    # batch size
    batch_size = embeddings_1.size(0)   

    # calculate the similarity score
    error = - torch.matmul(embeddings_1, embeddings_2.t())

    # get the similarity of the correct image-caption pairs (the diagonal of the similarity matrix)
    I = torch.autograd.Variable(dtype(torch.eye(batch_size).numpy()), requires_grad = True)
    diag = (error * I).sum(dim=0)
    
    # calculate the image to text and text to image cost.
    cost_1 = torch.clamp(.2 - error + diag, min = 0)
    cost_2 = torch.clamp(.2 - error + diag.view(-1, 1), min = 0)
    cost = cost_1 + cost_2
    
    # remove the diagonal for the cost matrix (i.e. count no costs for correct pairs)
    I_2 = torch.autograd.Variable(dtype(torch.eye(batch_size).numpy()), requires_grad = True)

    cost = (1 - I_2) * cost
    return cost.mean()

#implements the ordered embeddings loss function proposed by vendrov et all.
def ordered_loss(embeddings_1, embeddings_2, dtype):
    # batch size
    batch_size = embeddings_1.size(0)   

    # calculate the similarity score as described by vendrov et al. the partial order is image < caption, 
    # i.e. the captions are abstractions of the images (the wrong order results in worse results). 
    err = - (torch.clamp(embeddings_1 - embeddings_2[0], min = 0).norm(1, dim = 1, keepdim = True)**2)
    for x in range(1, embeddings_2.size(0)):
        e =  - (torch.clamp(embeddings_1 - embeddings_2[x], min = 0).norm(1, dim = 1, keepdim = True)**2)
        err = torch.cat((err, e), 1)

    # get the similarity of the correct image-caption pairs    
    I = torch.autograd.Variable(dtype(torch.eye(batch_size).numpy()), requires_grad = True)

    diag_1 = (err * I).sum(dim=0)

    # calculate the image to text and text to image cost.
    cost_1 = torch.clamp(.2 - diag_1 + err, min = 0)
    cost_2 = torch.clamp(.2 - diag_1.view(-1, 1) + err, min = 0)    
    cost = cost_1 + cost_2
    
    # remove the diagonal for the cost matrix (i.e. count no costs for correct pairs)
    I_2 = torch.autograd.Variable(dtype(torch.eye(batch_size).numpy()), requires_grad = True)
    
    cost = (1 - I_2) * cost
    
    return cost.mean()

# loss function forcing the weights of the attention heads, the resulting 
# attention matrices and the resulting embeddings to be different by a margin
def attention_loss(att_layer, attention, emb, margin = 1):
    # get the weight parameters from the attention layer
    weights = []
    for head in att_layer.att_heads:
        w = []
        for x in head.parameters():
            w.append(x)
        weights.append(w)
    loss = 0
    # calculate the loss of the weights
    for x in range(len(weights)):
        for y in range(x+1, len(weights)):
            loss += torch.clamp(margin - torch.norm(weights[x][0] - weights[y][0]), min = 0)
    # calculate the loss of the attention matrix
    for x in range(len(attention)):
        for y in range(x+1, len(attention)):
            loss += torch.clamp(margin - torch.norm(attention[x][0] - attention[y][0]), min = 0)
    # calculate the loss of the embeddings
    emb = emb.t().contiguous().view(len(weights), -1, emb.size(0))
    for x in range(emb.size(0)):
        for y in range(x+1, emb.size(0)):
            loss += torch.clamp(margin - torch.norm(emb[x] - emb[y]), min = 0)    
    return loss    