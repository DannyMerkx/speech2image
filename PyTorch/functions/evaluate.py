#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:36:08 2018

@author: danny

evaluation functions. contains only recall@n now. convenience functions for embedding
the data and calculating the recall@n to keep your NN training script clean
"""
import numpy as np
import torch
from torch.autograd import Variable

# class to evaluate image to caption models with mean and median rank and recall@n
class evaluate():
    def __init__(self, dtype, embed_function_1, embed_function_2):
        self.dtype = dtype
        self.embed_function_1 = embed_function_1
        self.embed_function_2 = embed_function_2
        
    # embed the captions and images
    def embed_data(self, iterator):
        # set to evaluation mode
        self.embed_function_1.eval()
        self.embed_function_2.eval()
        for batch in iterator:
            # load data and sort by caption length
            img, cap, lengths = batch
            sort = np.argsort(- np.array(lengths))
            cap = cap[sort]
            img = img[sort]
            lens = np.array(lengths)[sort]      
            # convert data to pytorch variables
            img, cap = Variable(self.dtype(img), requires_grad=False), Variable(self.dtype(cap),requires_grad=False)
            # embed the data
            img = self.embed_function_1(img)
            cap = self.embed_function_2(cap, lens)
            # reverse the sorting by length such that the data is in the same order for all 5 captions.
            cap = cap[torch.cuda.LongTensor(np.argsort(sort))]
            img = img[torch.cuda.LongTensor(np.argsort(sort))]
            # concat to existing tensor or create one if non-existent yet
            try:
                caption = torch.cat((caption, cap.data))
            except:
                caption = cap.data
            try:
                image = torch.cat((image, img.data))
            except:
                image = img.data
        # set the image and caption embeddings as class values.
        self.image_embeddings = image
        self.caption_embeddings = caption
        
    # creates the distance matrix that can then be input to the evaluation functions
    def dist_matrix(self, col = False, cosine = True):    
        # total number of the embeddings
        n_emb = self.image_embeddings.size()[0]
        # with the 5 captions per image we got 5 copies of all images, get rid of the copies.
        embeddings_1 = self.image_embeddings[0:n_emb//5, :]
        embeddings_2 = self.caption_embeddings
        # get the cosine similarity matrix for the embeddings by default. pass cosine = False to use the
        # ordered distance measure proposed by vendrov et al. 
        if cosine:
            sim = torch.matmul(embeddings_1, embeddings_2.t())
        else:
            sim = torch.clamp(embeddings_1 - embeddings_2[0], min = 0).norm(1, dim = 1, keepdim = True)**2
            for x in range(1, embeddings_2.size(0)):
                s = torch.clamp(embeddings_1 - embeddings_2[x], min = 0).norm(1, dim = 1, keepdim = True)**2
                sim = torch.cat((sim, s), 1)
            sim = - sim
            
        # the sorting dimension determines if we do cap2im or im2cap. Sorting the matrix by columns by setting col=True
        # results in image2caption
        if col:
            dim = 1
        else:
            dim = 0
        # apply sort two times to get a matrix where the values for each position indicate its rank in the column
        sorted, indices = sim.sort(dim, descending = True)
        sorted, indices = indices.sort(dim)

        # we now take the diagonal at intervals of n_emb /5 (i.e. the diagonal of each caption 'submatrix').
        # the ranks of the correct solutions are on the diagonal.
        diags = []
        for x in range(0,5):
            # we add one to each rank so we count from 1 instead of 0
            diag = indices.diag((int(x*(n_emb/5)))) + 1
            diags.append(diag.unsqueeze(1))
        # concat the diagonals
        self.ranks = torch.cat(diags,1)
    def median_rank(self, ranks):
        self.median = ranks.double().median()
    
    def mean_rank(self, ranks):
        self.mean = ranks.double().mean()
    def recall_at_n(self, ranks):
        if type(self.n) == int:
            r = ranks.le(self.n).double().mean()
        elif type(self.n) == list:
            r = []
            for x in self.n:
                r.append(ranks.le(x).double().mean())
        self.recall = r
    
    def caption2image(self):
        # calculate the recall and median rank
        self.dist_matrix(col = False)
        self.median_rank(self.ranks)
        self.mean_rank(self.ranks)
        self.recall_at_n(self.ranks)
    def image2caption(self):
        # calculate the recall and median rank
        self.dist_matrix(col = True)
        self.median_rank(self.ranks.min(1)[0])
        self.mean_rank(self.ranks.min(1)[0])
        self.recall_at_n(self.ranks.min(1)[0])
     

###############################################################################
    # functions used to return the class values
    def return_image_embeddings(self):
        return self.image_embeddings
    def return_caption_embeddings(self):
        return self.caption_embeddings
    def return_median_rank(self):
        return self.median
    def return_mean_rank(self):
        return self.mean
    def return_recall(self):
        return self.recall
###############################################################################
# functions to set some class values
    def set_n(self, n):
        # set n value for the recall at n
        self.n = n
    def set_image_embeddings(self, image_embeddings):
        # set image embeddings (e.g. if you want to calc recall for some pre-existing embeddings)
        self.image_embeddings = image_embeddings
    def set_caption_embeddings(self, caption_embeddings):
        self.caption_embeddings = caption_embeddings
    def set_embedder_1(self, embedder):
        # set a new model as embedder 1
        self.embed_function_1 = embedder
    def set_embedder_2(self, embedder):
        self.embed_function_2 = embedder
###############################################################################
    # function to run the image2caption or caption2 image and print the results
    def print_caption2image(self, prepend, epoch = 0):
        self.image2caption()
        r = 'recall :'
        for x in range(len(self.recall)):
            r += (' @' + str(self.n[x]) + ': ' + str(np.round(self.recall[x] * 100)),2)
        print(prepend + ' i2c,' + ' epoch:' + str(epoch) + ' ' + r + ' median: ' + str(self.median) + ' mean: ' + str(np.round(self.mean),2))  
    def print_image2caption(self, prepend, epoch = 0):
        self.caption2image()
        r = 'recall:'
        for x in range(len(self.recall)):
            r += (' @' + str(self.n[x]) + ': ' + str(np.round(self.recall[x] * 100)),2)
        print(prepend + ' c2i,' + ' epoch:' + str(epoch) + ' ' + r + ' median: ' + str(self.median) + ' mean: ' + str(np.round(self.mean),2))  