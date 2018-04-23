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

# hinge loss function to replace all those above, by using a list of bools to indicate which set of embeddings
# need to be normalised. Setting both to false results in the dot product, setting both to true
# results in the cosine similarity. normalising only the speech embeddings is what Harwath and Glass found
# worked best on their data.
def batch_hinge_loss(embeddings_1, embeddings_2, norm, cuda = True):
    # batch size
    batch_size = embeddings_1.size(0)   
    # calculate the norms of the embeddings and normalise the embeddings
    if norm[0]:
        embeddings_1 = embeddings_1 / embeddings_1.norm(2, dim = 1, keepdim = True)
    if norm[1]:
        embeddings_2 = embeddings_2 / embeddings_2.norm(2, dim = 1, keepdim = True)
    
    # calculate the similarity score
    error = - torch.matmul(embeddings_1, embeddings_2.t())

    # get the similarity of the correct image-caption pairs (the diagonal of the similarity matrix)
    I = torch.autograd.Variable(torch.eye(batch_size), requires_grad = True)
    if cuda:
        I.cuda()
    diag = (error * I).sum(dim=0)

    cost_1 = torch.clamp(0.2 - error + diag, min = 0)
    cost_2 = torch.clamp(0.2 - error + diag.view(-1, 1), min = 0)
    cost = cost_1 + cost_2
    
    I_2 = torch.autograd.Variable(torch.eye(batch_size), requires_grad = True)
    if cuda:
        I_2.cuda()
    
    cost = (1 - I_2) * cost

    return cost.mean()
