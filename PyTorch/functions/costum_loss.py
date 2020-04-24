#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:08:28 2018

@author: danny
Loss functions for image-caption retrieval
"""
import torch

# hinge loss function based on a symmetric distance measure. The loss function 
# uses the dot product, you get the cosine similarity by normalising your 
# embeddings at the output layer of the encoders. Optionally only use the top 
# n negative samples (neg_sample)
def batch_hinge_loss(embeddings_1, embeddings_2, dtype, neg_sample = False):
    # batch size
    batch_size = embeddings_1.size(0)   
    if neg_sample == False:
        neg_sample = batch_size
    # calculate the similarity score
    error = - torch.matmul(embeddings_1, embeddings_2.t())

    # get the similarity of the correct image-caption pairs 
    I = dtype(torch.eye(batch_size).numpy())
    diag = (error * I).sum(dim=0)
    
    # calculate the image to text and text to image cost. remove the diagonal
    # from the cost matrix (i.e. count no costs for correct pairs)
    I_2 = dtype(torch.eye(batch_size).numpy())
    cost_1 = torch.clamp(.2 - error + diag, min = 0)
    cost_1 = ((1 - I_2) * cost_1).sort(0)[0][-neg_sample:, :]
    cost_2 = torch.clamp(.2 - error + diag.view(-1, 1), min = 0)
    cost_2 = ((1 - I_2) * cost_2).sort(1)[0][:, -neg_sample:]

    cost = cost_1 + cost_2.t()
    return cost.mean()


#implements the ordered embeddings loss function proposed by vendrov et all. 
# Optionally only use the top n negative samples (neg_sample)
def ordered_loss(embeddings_1, embeddings_2, dtype, neg_sample = False):
    # batch size
    batch_size = embeddings_1.size(0)   
    if neg_sample == False:
        neg_sample = batch_size
    # calculate the similarity score as described by vendrov et al. the partial 
    # order is image < caption, i.e. the captions are abstractions of the 
    # images (the wrong order results in worse results). 
    err = (torch.clamp(torch.cat([embeddings_1 - x for x in embeddings_2]), 
                       min = 0).norm(1, dim = 1, keepdim = True)**2
           )
    err = err.reshape(batch_size,-1)

    # get the similarity of the correct image-caption pairs    
    I = dtype(torch.eye(batch_size).numpy())
    diag = (err * I).sum(dim=0)

    # calculate the image to text and text to image cost.
    I_2 = dtype(torch.eye(batch_size).numpy())
    cost_1 = torch.clamp(.05 - err + diag, min = 0)
    cost_1 = ((1 - I_2) * cost_1).sort(0)[0][-neg_sample:, :]

    cost_2 = torch.clamp(.05 - err + diag.view(-1, 1), min = 0)
    cost_2 = ((1 - I_2) * cost_2).sort(1)[0][:, -neg_sample:]

    cost = cost_1 + cost_2.t()

    return cost.mean()

###############################################################################
# loss function forcing the weights of the attention heads, the resulting 
# attention matrices and the resulting embeddings to be different by a margin
def attention_loss(att_layer, emb, margin = 1):
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
            loss += torch.clamp(margin - torch.norm(weights[x][0] - weights[y][0]), 
                                min = 0)
    # calculate the loss of the attention matrix
    for x in range(len(att_layer.alpha)):
        for y in range(x+1, len(att_layer.alpha)):
            loss += torch.clamp(margin - torch.norm(att_layer.alpha[x][0] - att_layer.alpha[y][0]), 
                                min = 0)
    # calculate the loss of the embeddings
    emb = emb.t().contiguous().view(len(weights), -1, emb.size(0))
    for x in range(emb.size(0)):
        for y in range(x+1, emb.size(0)):
            loss += torch.clamp(margin - torch.norm(emb[x] - emb[y]), min = 0)    
    return loss    
