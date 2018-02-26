#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:36:08 2018

@author: danny
"""
import sklearn.metrics.pairwise as sk
import numpy as np

def embed_data(iterator, embed_function_1, embed_function_2):
    speech = []
    image = []
    for batch in iterator:
        img, sp = batch
        sp = embed_function_1(sp)
        img = embed_function_2(img)
        speech.append(sp)
        image.append(img)
    return speech, image

def recall_at_n(embeddings_1, embeddings_2, n):
# calculate the recall at n for a retrieval task where given an embedding of some
# data we want to retrieve the embedding of a related piece of data (e.g. images and captions)
    # concatenate the embeddings (my embed data function delivers a list of batched embeddings)
    if len(embeddings_1) > 1:
        embeddings_1 = np.concatenate(embeddings_1)
    if len(embeddings_2) > 1:
        embeddings_2 = np.concatenate(embeddings_2)
        
    # get the cosine similarity matrix for the embeddings.
    sim = sk.cosine_similarity(embeddings_1, embeddings_2)
    # first the recall in the direction of embeddings_1 to embeddings_2
    # sort the columns (direction is now emb1 to emb2) of the sim matrix (negate the similarity 
    # matrix as argsort only works in ascending order). 
    # apply sort two times to get a matrix where the values for each position indicate its rank in the column
    sim_col_sorted = np.argsort(np.argsort(-sim, axis = 0), axis = 0)
    # the diagonal of the resulting matrix now holds the rank of the correct embedding pair
    if type(n) == int:
        recall = len([rank for rank in sim_col_sorted.diagonal() if rank < n])/len(sim_col_sorted.diagonal()) 
    if type(n) == list:
        recall = []
        for x in n:
            recall.append(len([rank for rank in sim_col_sorted.diagonal() if rank < x])/len(sim_col_sorted.diagonal())) 
    else: 
        recall = 'wrong type for parameter n, recall@n could not be calculated'
    # the average rank of the correct output
    avg_rank = np.mean(sim_col_sorted.diagonal()+1)
    return recall, avg_rank

def calc_recall_at_n(iterator, embed_function_1, embed_function_2, n):
    embeddings_1, embeddings_2 = embed_data(iterator, embed_function_1, embed_function_2)
    return recall_at_n(embeddings_1, embeddings_2, n)
    