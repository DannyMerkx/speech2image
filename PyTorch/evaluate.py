#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:36:08 2018

@author: danny

evaluation functions. contains only recall@n now. convenience functions for embedding
the data and calculating the recall@n to keep your NN training script clean
"""
import sklearn.metrics.pairwise as sk
import numpy as np
from torch.autograd import Variable
# embeds the validation or test data using the trained neural network. Takes
# an iterator (minibatcher) and the embedding functions (i.e. deterministic 
# output functions for your network). 
def embed_data(iterator, embed_function_1, embed_function_2, dtype):
    speech = []
    image = []
    # set to evaluation mode
    embed_function_1.eval()
    embed_function_2.eval()
    for batch in iterator:
        img, sp = batch
        # convert data to pytorch variables
        img, sp = Variable(dtype(img), requires_grad=False), Variable(dtype(sp),requires_grad=False)
        sp = embed_function_1(sp)
        img = embed_function_2(img)
        speech.append((sp.data).cpu().numpy())
        image.append((img.data).cpu().numpy())
    return speech, image

# returns the recall@n over your test or validation set. Takes the embeddings 
# returned by embed_data and n. n can be a scalar or a array so you can calculate
# recall for multiple n's 
def recall_at_n(embeddings_1, embeddings_2, n):
# calculate the recall at n for a retrieval task where given an embedding of some
# data we want to retrieve the embedding of a related piece of data (e.g. images and captions)
    
    # concatenate the embeddings (the embed data function delivers a list of batches)

    if len(embeddings_1) > 1:
        embeddings_1 = np.concatenate(embeddings_1)
    else:
        embeddings_1 = embeddings_1[0]
    if len(embeddings_2) > 1:
        embeddings_2 = np.concatenate(embeddings_2)
    else:
        embeddings_2 = embeddings_2[0]
        
    # get the cosine similarity matrix for the embeddings.
    sim = sk.cosine_similarity(embeddings_1, embeddings_2)
    
    #recall can be calculated in 2 directions e.g. speech to image and image to speech
    
    # sort the columns (direction is now embeddings_1 to embeddings_2) of the sim matrix (negate the similarity 
    # matrix as argsort works in ascending order) 
    # apply sort two times to get a matrix where the values for each position indicate its rank in the column
    sim_col_sorted = np.argsort(np.argsort(-sim, axis = 0), axis = 0)
    # the diagonal of the resulting matrix now holds the rank of the correct embedding pair
    if type(n) == int:
        recall_col = len([rank for rank in sim_col_sorted.diagonal() if rank < n])/len(sim_col_sorted.diagonal()) 
    if type(n) == list:
        recall_col = []
        for x in n:
            recall_col.append(len([rank for rank in sim_col_sorted.diagonal() if rank < x])/len(sim_col_sorted.diagonal()))    
    else: 
        recall_col = 'wrong type for parameter n, recall@n could not be calculated'
    # the average rank of the correct output
    avg_rank_col = np.mean(sim_col_sorted.diagonal()+1)
    
    # sort by row, the retrieval direction is now embeddings_2 to embeddings_1
    sim_row_sorted = np.argsort(np.argsort(-sim, axis = 1), axis = 1)
    # the diagonal of the resulting matrix now holds the rank of the correct embedding pair
    if type(n) == int:
        recall_row = len([rank for rank in sim_row_sorted.diagonal() if rank < n])/len(sim_row_sorted.diagonal()) 
    if type(n) == list:
        recall_row = []
        for x in n:
            recall_row.append(len([rank for rank in sim_row_sorted.diagonal() if rank < x])/len(sim_row_sorted.diagonal()))     
    else: 
        recall_row = 'wrong type for parameter n, recall@n could not be calculated'
    # the average rank of the correct output
    avg_rank_row = np.mean(sim_row_sorted.diagonal()+1)
    
    return recall_col, avg_rank_col, recall_row, avg_rank_row

# small convenience function for combining everything in this script
def calc_recall_at_n(iterator, embed_function_1, embed_function_2, n, dtype):
    embeddings_1, embeddings_2 = embed_data(iterator, embed_function_1, embed_function_2, dtype)
    return recall_at_n(embeddings_1, embeddings_2, n)
    
