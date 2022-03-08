#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:36:08 2018

@author: danny

evaluation functions. five fold r@n is meant for mscoco: mscoco has 5k test
items but other papers always report 

"""
import numpy as np
import torch

# class to evaluate c2i models with mean and median rank and recall@n
class evaluate():
    def __init__(self, dtype, embed_function_1, embed_function_2, test_size = 1000):
        self.dtype = dtype
        self.embed_function_1 = embed_function_1
        self.embed_function_2 = embed_function_2
	# size of the test/eval set, default is 1000 (appropriate for flickr and places) 	
        self.test_size = test_size
        # set dist to cosine by default
        self.dist = self.cosine
    # embed the captions and images (usefull if not running a test epoch
    # before calculating R@N for some reason)
    def embed_data(self, iterator):
        # set to evaluation mode
        self.embed_function_1.eval()
        self.embed_function_2.eval()
        caption = self.dtype()
        image = self.dtype()
        for batch in iterator:
            img, cap, lengths = batch    
            # convert data to the right pytorch type and disable gradients
            img, cap = self.dtype(img), self.dtype(cap)
            img.requires_grad = False
            cap.requires_grad = False
            # embed the data
            img = self.embed_function_1(img)
            cap = self.embed_function_2(cap, lengths)

            caption = torch.cat((caption, cap.data))
            image = torch.cat((image, img.data))
        # set the image and caption embeddings as class values.
        self.image_embeddings = image
        self.caption_embeddings = caption
         
    # distance functions for calculating recall
    def cosine(self, emb_1, emb_2):
        # cosine expects embeddings to be normalised in the embedder
        return torch.matmul(emb_1, emb_2.t())
    def ordered(self, emb_1, emb_2):
        return  - torch.clamp(emb_1 - emb_2, min = 0).norm(1, dim = 1, 
                                                           keepdim = True)**2
    # calculate caption2image
    def c2i(self):
        # total number of the embeddings
        n_emb = self.image_embeddings.size()[0]
	
        embeddings_1 = self.caption_embeddings
        # if we get 5 captions per image (e.g. flickr) we got 5 copies of each image embedding 
        # get rid of the copies.
        embeddings_2 = self.image_embeddings
        if self.test_size != n_emb:
            embeddings_2 = self.image_embeddings[0:self.test_size, :]   
        ranks = []
        for index, emb in enumerate(embeddings_1):
            sim = self.dist(emb, embeddings_2)
            # apply sort two times to get a tensor where the values for each 
            # image indicates the rank of its distance to the caption
            sorted, indices = sim.sort(descending = True)
            sorted, indices = indices.sort()
            # add 1 to the rank so that the recall measure has 1 as best rank
            rank = indices[np.mod(index, embeddings_2.size()[0])] + 1
            ranks.append(rank)
            
        self.ranks = self.dtype(ranks)
    # calculate image2caption, n is the number of captions per image
    def i2c(self, n = 5):
        # total number of embeddings
        n_emb = self.image_embeddings.size()[0]
        # if we get 5 captions per image (e.g. flickr) we got 5 copies of each image embedding 
        # get rid of the copies.
        embeddings_1 = self.image_embeddings
        if self.test_size != n_emb:
            embeddings_1 = self.image_embeddings[0:self.test_size, :]
        embeddings_2 = self.caption_embeddings
        # calculate the distance for 1 img embedding at a time to save space
        ranks = []        
        for index, emb in enumerate(embeddings_1):
            sim = self.dist(emb, embeddings_2)
            # apply sort two times to get a tensor where the values for each 
            # caption indicates the rank of its distance to the image
            sorted, indices = sim.sort(descending = True)
            sorted, indices = indices.sort()
            # extract the ranks of all 5 captions and append them to the total
            if self.test_size != n_emb:
                inds = [index + (x * (n_emb // n)) for x in range (n)]
                ranks.append(indices[inds].unsqueeze(1) + 1)
            else:
                rank = indices[np.mod(index, embeddings_2.size()[0])] + 1
                ranks.append(rank)
        if self.test_size != n_emb:
            # create a 5 by n_images matrix with the rank of each correct caption
            ranks = torch.cat(ranks, 1)
            self.ranks = ranks.min(0)[0]
        else:
            self.ranks = self.dtype(ranks)

    # calculate median rank            
    def median_rank(self, ranks):
        self.median = ranks.double().median().cpu().numpy()
    # calculate mean rank
    def mean_rank(self, ranks):
        self.mean = ranks.double().mean().cpu().numpy()
    # extract r@n from the ranks by taking all ranks less than or equal to n
    # as a portion of total ranks
    def recall_at_n(self, ranks):
        if type(self.n) == int:
            r = ranks.le(self.n).double().mean().cpu().numpy()
        elif type(self.n) == list:
            r = []
            for x in self.n:
                r.append(ranks.le(x).double().mean().cpu().numpy())
        self.recall = r
    # combine all the function calls for convenience
    def caption2image(self):
        self.c2i()
        self.median_rank(self.ranks)
        self.mean_rank(self.ranks)
        self.recall_at_n(self.ranks)
    def image2caption(self):
        self.i2c()
        # take only the caption with the lowest rank into consideration 
        self.median_rank(self.ranks)
        self.mean_rank(self.ranks)
        self.recall_at_n(self.ranks)

###############################################################################
# functions to set some class values
    def set_n(self, n):
        # set n value for the recall at n
        self.n = n
    def set_image_embeddings(self, image_embeddings):
        # set image embeddings (e.g. if you want to calc recall for some 
        # pre-existing embeddings)
        self.image_embeddings = image_embeddings
    def set_caption_embeddings(self, caption_embeddings):
        self.caption_embeddings = caption_embeddings
    def set_embedder_1(self, embedder):
        # set a new model as embedder 1
        self.embed_function_1 = embedder
    def set_embedder_2(self, embedder):
        # set a new model as embedder 2
        self.embed_function_2 = embedder
    def set_cosine(self):
        # set the distance function for recall to cosine
        self.dist = self.cosine
    def set_ordered(self):
        # set the distance function to ordered
        self.dist = self.ordered
###############################################################################
    # functions to run the image2caption or caption2image and print the results
    def print_caption2image(self, prepend, epoch = 0):
        self.image2caption()
        r_msg = 'recall :'
        for x in range(len(self.recall)):
            r = np.round(self.recall[x] * 100, 2)
            r_msg += (f' @{self.n[x]}: {r}')
        mean = np.round(self.mean, 2)
        print(f'{prepend} i2c, epoch: {epoch} {r_msg} median: {self.median} mean: {mean}')  
    def print_image2caption(self, prepend, epoch = 0):
        self.caption2image()
        r_msg = 'recall :'
        for x in range(len(self.recall)):
            r = np.round(self.recall[x] * 100, 2)
            r_msg += (f' @{self.n[x]}: {r}')
        mean = np.round(self.mean, 2)
        print(f'{prepend} c2i, epoch: {epoch} {r_msg} median: {self.median} mean: {mean}')  
