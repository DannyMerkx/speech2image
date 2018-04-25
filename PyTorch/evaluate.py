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
import torch
from torch.autograd import Variable

# small convenience functions for combining everything in this script.
# N.B. make sure the order in which you pass the embedding functions is the 
# order in which the iterator yields the appropriate features!

def image2speech(iterator, image_embed_function, speech_embed_function, n, dtype):
    im_embeddings, speech_embeddings = embed_data(iterator, image_embed_function, speech_embed_function, dtype)
    return recall_at_n(im_embeddings, speech_embeddings, n)

def speech2image(iterator, image_embed_function, speech_embed_function, n, dtype):
    im_embeddings, speech_embeddings = embed_data(iterator, image_embed_function, speech_embed_function, dtype)
    return recall_at_n(speech_embeddings, im_embeddings, n)  

# embeds the validation or test data using the trained neural network. Takes
# an iterator (minibatcher) and the embedding functions (i.e. deterministic 
# output functions for your network). 
def embed_data(iterator, embed_function_1, embed_function_2, dtype):
    # set to evaluation mode
    embed_function_1.eval()
    embed_function_2.eval()
    for batch in iterator:
        img, sp = batch
        # convert data to pytorch variables
        img, sp = Variable(dtype(img), requires_grad=False), Variable(dtype(sp),requires_grad=False)
        # embed the data
        img = embed_function_1(img)
        sp = embed_function_2(sp)
        # concat to existing tensor or create one if non-existent yet
        try:
            speech = torch.cat((speech, sp.data))
        except:
            speech = sp.data
        try:
            image = torch.cat((image, img.data))
        except:
            image = img.data
    return speech, image

###########################################################################################

def recall_at_n(embeddings_1, embeddings_2, n):
# calculate the recall at n for a retrieval task where given an embedding of some
# data we want to retrieve the embedding of a related piece of data (e.g. images and captions)
# recall works in the direction of embeddings_1 to embeddings_2, so if you pass images, and speech
# respectively you calculate image to speech retrieval scores.
    
    # get the cosine similarity matrix for the embeddings.
    sim = torch.matmul(embeddings_1, embeddings_2.t())
    # apply sort two times to get a matrix where the values for each position indicate its rank in the column
    sorted, indices = sim.sort(dim = 1, descending = True)
    sorted, indices = indices.sort(dim = 1)
    # the diagonal of the resulting matrix diagonal now holds the rank of the correct embedding pair (add 1 cause 
    # sort counts from 0, we want the top rank to be indexed as 1)
    diag = indices.diag() +1
    if type(n) == int:
        recall = diag.le(n).double().mean()
    elif type(n) == list:
        recall = []
        for x in n:
            recall.append(diag.le(x).double().mean())    
    # the average rank of the correct output
    median_rank = diag.median()
    return(recall, median_rank)