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

def image2speech(iterator, image_embed_function, speech_embed_function, n, dtype, mode ='full'):
    im_embeddings, speech_embeddings = embed_data(iterator, image_embed_function, speech_embed_function, dtype)
    return recall_at_n(im_embeddings, speech_embeddings, n, mode)

def speech2image(iterator, image_embed_function, speech_embed_function, n, dtype, mode = 'full'):
    im_embeddings, speech_embeddings = embed_data(iterator, image_embed_function, speech_embed_function, dtype)
    return recall_at_n(speech_embeddings, im_embeddings, n, mode)  

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
        # retrieve data back from the gpu to the cpu if applicable and convert back to numpy data
        try:
            speech.cat((speech, sp))
        except:
            speech = sp
        try:
            image.cat((image, img))
        except:
            image = img
    return speech, image

###########################################################################################
# returns the recall@n over your test or validation set. Takes the embeddings 
# returned by embed_data and n. n can be a scalar or a array so you can calculate
# recall for multiple n's , mode indicates one of several modes, the default full
# mode is fast but calculates an n by m similarity matrix which might become to big 
# for memory in larger datasets
def recall_at_n(embeddings_1, embeddings_2, n, mode):
# calculate the recall at n for a retrieval task where given an embedding of some
# data we want to retrieve the embedding of a related piece of data (e.g. images and captions)
# recall works in the direction of embeddings_1 to embeddings_2, so if you pass images, and speech
# respectively you calculate image to speech retrieval scores.

    # fastest way to calculate recall, but it creates a full similarity matrix for all the embeddings and 
    # might lead to memory errors in bigger datasets (testing with 6gb of ram showed this was possible
    # for 2^13 or about 8000 embedding pairs)
    if mode == 'full':
        # get the cosine similarity matrix for the embeddings.
        sim = torch.matmul(embeddings_1, embeddings_2.t())
        # apply sort two times to get a matrix where the values for each position indicate its rank in the column
        sorted, indices = sim.sort(dim = 1, descending = True)
        sorted, indices = indices.sort(dim = 1)
        # the diagonal of the resulting matrix diagonal now holds the rank of the correct embedding pair (add 1 cause 
        # sort counts from 0, we want the top rank to be indexed as 1)
        diag = indices.diag() +1
        if type(n) == int:
            recall = diag.le(n).double().mean().data.numpy()
        elif type(n) == list:
            recall = []
            for x in n:
                recall.append(diag.le(x).double().mean().data.numpy())    
        # the average rank of the correct output
        median_rank = diag.median().data.numpy()
    
        return(recall, median_rank)
    # slower than full mode, but calculates a similarity array instead of matrix for one embedding vs all 
    # embeddings in embeddings_2 which is less memory intensive
    elif mode == 'array':
        # keep track of a top n similarity scores based on the required recall@n        
        if type(n) == int:
            recall = 0
        elif type(n) == list:
            recall = np.zeros(len(n))
        avg_rank = 0
        for index_1 in range(0, len(embeddings_1)):
            # calculate the similatiry between one sample from embeddings_1 and all embeddings in embeddings_2
            sim = -sk.cosine_similarity(embeddings_1[index_1], embeddings_2)
            # double argsorting such that the array holds the rank of each embedding in embeddings_2  
            sim = sim.argsort().argsort()
            # keep track of the average rank of the corret embedding pair (add 1 to the rank because python counts from 0)
            avg_rank += sim[0][index_1]+1 
            if type(n) == int:
                if sim[0][index_1] < n: 
                    recall += 1
            elif type(n) == list:
                for x in range(0,len(n)):
                    if sim[0][index_1] < n[x]:
                        recall[x] += 1
        recall = recall / len(embeddings_1)
        avg_rank = avg_rank / len(embeddings_1)
        return(recall, avg_rank)
    
    # very slow but used almost no memory as similarity is calculated on a sample by sample basis. Not recommended
    # unless necessary      
    if mode == 'big':
        # keep track of a top n similarity scores based on the required recall@n
        if type(n) == int:
            recall = 0
        elif type(n) == list:
            recall = np.zeros(len(n))
        avg_rank = 0 
        for index_1 in range(0, len(embeddings_1)):
            top_n = np.zeros([n[-1],2])
            # keep track of the similarity score for the correct pair so we can calculate the average rank
            # of the correct embedding pair
            correct_embed = float(-sk.cosine_similarity(embeddings_1[index_1], embeddings_2[index_1]))
            correct_rank = 1
            for index_2 in range(0,len(embeddings_2)):
                # calculate the similarity between each pair of embeddings
                sim = float(-sk.cosine_similarity(embeddings_1[index_1], embeddings_2[index_2]))
                if sim < correct_embed:
                    correct_rank +=1
                # add the new similarity score and the index of the retrieved embedding to the top n
                top_n = np.concatenate((top_n, np.matrix([sim, index_2])))       
                # sort the top n and keep and throw away the lowest score
                top_n = top_n[np.array(top_n.argsort(axis = 0)[:n[-1],0]).reshape(-1)]
            # now for each number of n we want to calculate the recall for check if the index of embedding_1 (x)
            # is in the top_n 
            if type(n) == int:
                if index_1 in top_n[:n[x],1]:    
                    recall += 1
            elif type(n) == list:
                for x in range(0,len(n)):
                    if index_1 in top_n[:n[x],1]:
                        recall[x] += 1
            avg_rank += correct_rank
        recall = recall / len(embeddings_1)
        avg_rank = avg_rank / len(embeddings_1)
        return recall, avg_rank
    
    # As an alternative to the slower modes for big datasets, implement an 
    # approximation where recall is based on a feasibly sized subsample of the dataset. 

    
