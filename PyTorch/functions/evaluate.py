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
        self.dist = self.cosine
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
    # distance functions for calculating recall
    def cosine(self, emb_1, emb_2):
        return torch.matmul(emb_1, emb_2.t())
    def ordered(self, emb_1, emb_2):
        return  - torch.clamp(emb_1 - emb_2, min = 0).norm(1, dim = 1, keepdim = True)**2
        
    def c2i(self):
        # total number of the embeddings
        n_emb = self.image_embeddings.size()[0]
        # with the 5 captions per image we got 5 copies of all images, get rid of the copies.
        embeddings_1 = self.caption_embeddings
        embeddings_2 = self.image_embeddings[0:n_emb//5, :]
        # get the cosine similarity matrix for the embeddings by default. pass cosine = False to use the
        # ordered distance measure proposed by vendrov et al. 
        ranks = []
        for index, emb in enumerate(embeddings_1):
            sim = self.dist(emb, embeddings_2)
            # apply sort two times to get a matrix where the values for each position indicate its rank in the column
            sorted, indices = sim.sort(descending = True)
            sorted, indices = indices.sort()

            rank = indices[np.mod(index, embeddings_2.size()[0])] + 1
            ranks.append(rank)

        # concat the diagonals
        self.ranks = self.dtype(ranks)        

    def i2c(self):
        # total number of the embeddings
        n_emb = self.image_embeddings.size()[0]
        # with the 5 captions per image we got 5 copies of all images, get rid of the copies.
        embeddings_1 = self.image_embeddings[0:n_emb//5, :]
        embeddings_2 = self.caption_embeddings
        # get the cosine similarity matrix for the embeddings by default. pass cosine = False to use the
        # ordered distance measure proposed by vendrov et al. 
        ranks = []
        for index, emb in enumerate(embeddings_1):
            sim = self.dist(emb, embeddings_2)
            # apply sort two times to get a matrix where the values for each position indicate its rank in the column
            sorted, indices = sim.sort(descending = True)
            sorted, indices = indices.sort()
            # get the indices of the 5 captions 
            inds = [index + (x * (n_emb // 5)) for x in range (5)]
            ranks.append(indices[inds].unsqueeze(1) + 1)

        # concat the diagonals
        self.ranks = torch.cat(ranks,1)
                
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
        self.c2i()
        self.median_rank(self.ranks)
        self.mean_rank(self.ranks)
        self.recall_at_n(self.ranks)
    def image2caption(self):
        # calculate the recall and median rank
        self.i2c()
        self.median_rank(self.ranks.min(0)[0])
        self.mean_rank(self.ranks.min(0)[0])
        self.recall_at_n(self.ranks.min(0)[0])
    # functions to run caption2image and image2caption on a 5 fold test set.
    # creates 5 random folds and accumulates, averages and prints the results 
    def fivefold_c2i(self, prepend, epoch = 0):
        # get the embeddings for the full test set
        capts = (self.caption_embeddings)
        imgs = (self.image_embeddings)
        # get indices for the full test set, shuffle them and reshape them to
        # create 5 random folds
        x = np.array([x for x in range(5000)])
        np.random.shuffle(x)
        x = np.reshape(x, (5,1000))
        # variables to save the evaluation metrics 
        median_rank = []
        mean_rank = []
        recall = []
        for y in range(5):
            # for the current fold get the indices of the embeddings. Add 5 increments of 5000 to the indices
            # in order to retrieve all 5 captions for each image in the fold.
            fold = torch.cuda.LongTensor(np.concatenate([x[y] + z  for z in range(0,25000,5000)]))
            # overwrite the embeddings variables with the current fold
            self.set_caption_embeddings(capts[fold])
            self.set_image_embeddings(imgs[fold])
            # perform the caption2image calculations
            self.caption2image()
            # retrieve the calculated metrics
            median_rank.append(self.median)
            mean_rank.append(self.mean)
            recall.append(self.recall)
        # calculate the average metrics over all folds
        self.median = torch.FloatTensor(median_rank).mean()
        self.mean = torch.FloatTensor(mean_rank).mean()
        self.recall = torch.FloatTensor(recall).mean(0)
        # reset the embedding variables to the full set
        self.set_caption_embeddings(capts)
        self.set_image_embeddings(imgs)
        # print the scores
        r = 'recall :'
        for x in range(len(self.recall)):
            r += (' @' + str(self.n[x]) + ': ' + str(np.round(self.recall[x] * 100,2)))
        print(prepend + ' c2i,' + ' epoch:' + str(epoch) + ' ' + r + ' median: ' + str(self.median) + ' mean: ' + str(np.round(self.mean,2)))  

    def fivefold_i2c(self, prepend, epoch = 0):
        # get the embeddings for the full test set
        capts = (self.caption_embeddings)
        imgs = (self.image_embeddings)
        # get indices for the full test set, shuffle them and reshape them to
        # create 5 random folds
        x = np.array([x for x in range(5000)])
        np.random.shuffle(x)
        x = np.reshape(x, (5,1000))
        # variables to save the evaluation metrics 
        median_rank = []
        mean_rank = []
        recall = []
        for y in range(5):
            # for the current fold get the indices of the embeddings. Add 5 increments of 5000 to the indices
            # in order to retrieve all 5 captions for each image in the fold.
            fold = torch.cuda.LongTensor(np.concatenate([x[y] + z  for z in range(0,25000,5000)]))
            # overwrite the embeddings variables with the current fold
            self.set_caption_embeddings(capts[fold])
            self.set_image_embeddings(imgs[fold])
            # perform the image2caption calculations
            self.image2caption()
            # retrieve the calculated metrics
            median_rank.append(self.median)
            mean_rank.append(self.mean)
            recall.append(self.recall)
        # calculate the average metrics over all folds
        self.median = torch.FloatTensor(median_rank).mean()
        self.mean = torch.FloatTensor(mean_rank).mean()
        self.recall = torch.FloatTensor(recall).mean(0)
        # reset the embedding variables to the full set
        self.set_caption_embeddings(capts)
        self.set_image_embeddings(imgs)
        r = 'recall :'
        for x in range(len(self.recall)):
            r += (' @' + str(self.n[x]) + ': ' + str(np.round(self.recall[x] * 100,2)))
        print(prepend + ' i2c,' + ' epoch:' + str(epoch) + ' ' + r + ' median: ' + str(self.median) + ' mean: ' + str(np.round(self.mean,2)))  

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
    def set_cosine(self):
        # set the distance function for recall to cosine
        self.dist = self.cosine
    def set_ordered(self):
        # set the distance function to ordered
        self.dist = self.ordered
###############################################################################
    # function to run the image2caption or caption2 image and print the results
    def print_caption2image(self, prepend, epoch = 0):
        self.image2caption()
        r = 'recall :'
        for x in range(len(self.recall)):
            r += (' @' + str(self.n[x]) + ': ' + str(np.round(self.recall[x] * 100,2)))
        print(prepend + ' i2c,' + ' epoch:' + str(epoch) + ' ' + r + ' median: ' + str(self.median) + ' mean: ' + str(np.round(self.mean,2)))  
    def print_image2caption(self, prepend, epoch = 0):
        self.caption2image()
        r = 'recall:'
        for x in range(len(self.recall)):
            r += (' @' + str(self.n[x]) + ': ' + str(np.round(self.recall[x] * 100,2)))
        print(prepend + ' c2i,' + ' epoch:' + str(epoch) + ' ' + r + ' median: ' + str(self.median) + ' mean: ' + str(np.round(self.mean,2)))  