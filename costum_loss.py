#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:08:28 2018

@author: danny
"""
import theano.tensor as T
# batch hinge loss function using dot product similarity measure
def dot_hinge_loss(embeddings_1, embeddings_2):
    # batch size
    batch_size = embeddings_1.shape[0]
    # calculate the dot product
    dot_prod = T.dot(embeddings_1, embeddings_2.T)
    
    # get the similarity of the correct image-caption pairs (the diagonal of the similarity matrix)
    matched = dot_prod.diagonal()
    # get the average mismatch of the image with incorrect captions
    # sum the matrix along the corresponding axis, correct for including the correct pair, divide by batch size -1
    # (also to correct for including the correct pair)
    mismatch_1 = (dot_prod.sum(axis=0) - dot_prod.diagonal() ) /(batch_size-1)
    # get the average mismatch of the captions with incorrect images
    mismatch_2 = (dot_prod.sum(axis=1) - dot_prod.diagonal() ) /(batch_size-1)

    return T.sum(T.maximum(0, mismatch_1 - matched+1) + T.maximum(0, mismatch_2 - matched+1))

# hinge loss using cosine similarity
def cosine_hinge_loss(embeddings_1, embeddings_2):
    # batch size
    batch_size = embeddings_1.shape[0]
    # calculate the numerator
    numerator = T.dot(embeddings_1,embeddings_2.T)
    # calculate the denominator
    denom1 = T.sum(T.sqr(embeddings_1),axis=1)
    denom2 = T.sum(T.sqr(embeddings_2),axis=1)
    
    denominator = T.sqrt(T.tensordot(denom1, denom2, axes = 0))
    # similarity matrix
    sim = numerator/denominator
     # get the similarity of the correct image-caption pairs (the diagonal of the similarity matrix)
    matched = sim.diagonal()
    # get the average mismatch of the image with incorrect captions
    # sum the matrix along the corresponding axis, correct for including the correct pair, divide by batch size -1
    # (also to correct for including the correct pair)
    mismatch_1 = (sim.sum(axis=0) - sim.diagonal() ) /(batch_size-1)
    # get the average mismatch of the captions with incorrect images
    mismatch_2 = (sim.sum(axis=1) - sim.diagonal() ) /(batch_size-1)

    return T.sum(T.maximum(0, mismatch_1 - matched+1) + T.maximum(0, mismatch_2 - matched+1))

# hinge loss where only one set of embeddings is l2 normalised like in the original
# harwath and glass paper (l2 normalise of speech embeddings). uncomment the appropriate 
# line to switch which embeddings to normalise, normalising both results in the cosine hinge loss of course. 
def l2norm_hinge_loss(embeddings_1, embeddings_2):
    # batch size
    batch_size = embeddings_1.shape[0]   
    # calculate the norms of the embeddings
    
    #denom1 = T.sqrt(T.sum(T.sqr(embeddings_1),axis=1))
    denom2 = T.sqrt(T.sum(T.sqr(embeddings_2),axis=1))   
    
    # calculate the similarity score and normalise one or both embeddings
    sim = T.dot(embeddings_1, embeddings_2.T/denom2)
    #sim = T.dot((embeddings_1.T/denom1).T, embeddings_2.T)
    #sim = T.dot((embeddings_1.T/denom1).T, embeddings_2.T/denom2)
    
    # get the similarity of the correct image-caption pairs (the diagonal of the similarity matrix)
    matched = sim.diagonal()
    # get the average mismatch of the image with incorrect captions
    # sum the matrix along the corresponding axis, correct for including the correct pair, divide by batch size -1
    # (also to correct for including the correct pair)
    mismatch_1 = (sim.sum(axis=0) - sim.diagonal() ) /(batch_size-1)
    # get the average mismatch of the captions with incorrect images
    mismatch_2 = (sim.sum(axis=1) - sim.diagonal() ) /(batch_size-1)

    return T.sum(T.maximum(0, mismatch_1 - matched+1) + T.maximum(0, mismatch_2 - matched+1))