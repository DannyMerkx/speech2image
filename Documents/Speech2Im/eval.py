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
def recall_at_n(speech_embeddings, image_embeddings, n):
    if len(image_embeddings) > 1:
        speech_embeddings = np.concatenate(speech_embeddings)
        image_embeddings = np.concatenate(image_embeddings)
    # get the cosine similarity matrix for the embeddings.
    sim = sk.cosine_similarity(speech_embeddings, image_embeddings)
    #
    speech2im = np.argsort(sim, axis = 0)
    recall = len([x for x in speech2im.diagonal() if x < n])/len(speech2im.diagonal()) 
    
def calc_recall_at_n(iterator, embed_function_1, embed_function_2, n):
    embeddings_1, embeddings_2 = embed_data(iterator, embed_function_1, embed_function_2)
    recall = recall_at_n(embeddings_1, embeddings_2, n)