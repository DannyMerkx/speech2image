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

# small convenience functions for combining everything in this script.
# N.B. make sure the order in which you pass the embedding functions is the 
# order in which the iterator yields the appropriate features!

def image2speech(iterator, image_embed_function, speech_embed_function, n, mode ='full'):
    im_embeddings, speech_embeddings = embed_data(iterator, image_embed_function, speech_embed_function)
    return recall_at_n(im_embeddings, speech_embeddings, n, mode)

def speech2image(iterator, image_embed_function, speech_embed_function, n, mode = 'full'):
    im_embeddings, speech_embeddings = embed_data(iterator, image_embed_function, speech_embed_function)
    return recall_at_n(speech_embeddings, im_embeddings, n, mode)  

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
        # embed the data
        sp = embed_function_1(sp)
        img = embed_function_2(img)
        # retrieve data back from the gpu to the cpu if applicable and convert back to numpy data
        speech.append((sp.data).cpu().numpy())
        image.append((img.data).cpu().numpy())
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
    
    # concatenate the embeddings (the embed data function delivers a list of batches)
    if len(embeddings_1) > 1:
        embeddings_1 = np.matrix(np.concatenate(embeddings_1))
    else:
        embeddings_1 = np.matrix(embeddings_1[0])
    if len(embeddings_2) > 1:
        embeddings_2 = np.matrix(np.concatenate(embeddings_2))
    else:
        embeddings_2 = np.matrix(embeddings_2[0])

    # fastest way to calculate recall, but it creates a full similarity matrix for all the embeddings and 
    # might lead to memory errors in bigger datasets (testing with 6gb of ram showed this was possible
    # for 2^13 or about 8000 embedding pairs)
    if mode == 'full':
        # get the cosine similarity matrix for the embeddings.
        sim = sk.cosine_similarity(embeddings_1, embeddings_2)
        # apply sort two times to get a matrix where the values for each position indicate its rank in the column
        # similarity is negated because argsort works in ascending order
        sim_col_sorted = np.argsort(np.argsort(-sim, axis = 1), axis = 1)
        # the diagonal of the resulting matrix diagonal now holds the rank of the correct embedding pair
        if type(n) == int:
            recall = len([rank for rank in sim_col_sorted.diagonal() if rank < n])/len(sim_col_sorted.diagonal()) 
        elif type(n) == list:
            recall = []
            for x in n:
                recall.append(len([rank for rank in sim_col_sorted.diagonal() if rank < x])/len(sim_col_sorted.diagonal()))    
        # the average rank of the correct output
        avg_rank = np.mean(sim_col_sorted.diagonal()+1)
    
        return(recall, avg_rank)
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

    
