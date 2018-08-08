#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:45:31 2018

@author: danny
"""
from minibatchers import iterate_tokens_5fold, iterate_raw_text_5fold, iterate_audio_5fold, iterate_snli_tokens, iterate_snli
import numpy as np
from torch.autograd import Variable
import torch
import os

class flickr_trainer():
    def __init__(self, img_embedder, cap_embedder, optimizer, loss, vis, cap):
        # default datatype, change to cuda by calling set_cuda
        self.dtype = torch.FloatTensor
        # set the embedders optimizer and loss function. set an empty scheduler to keep lr scheduling optional.
        self.img_embedder = img_embedder
        self.cap_embedder = cap_embedder
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = []
        # names of the features to be loaded by the batcher
        self.vis = vis
        self.cap = cap
        # keep track of an iteration for lr scheduling
        self.iteration = 0
        # set default for the caption length
        self.max_len = 300
    # the possible minibatcher for all different types of data for the flickr database
    def token_batcher(self, data, batch_size, shuffle):
        return iterate_tokens_5fold(data, batch_size, self.vis, self.cap, self.dict_loc, self.max_len, shuffle)
    def audio_batcher(self, data, batch_size, shuffle):
        return iterate_audio_5fold(data, batch_size, self.vis, self.cap, self.max_len, shuffle)
    def raw_text_batcher(self, data, batch_size, shuffle):
        return iterate_raw_text_5fold(data, batch_size, self.vis, self.cap, self.max_len, shuffle)      
    # functions to set which minibatcher to use. Needs to be called as no default is set.
    def set_token_batcher(self):
        self.batcher = self.token_batcher
    def set_raw_text_batcher(self):
        self.batcher = self.raw_text_batcher
    def set_audio_batcher(self):
        self.batcher = self.audio_batcher
    # function to set the max frame/word/character length of the captions to non default value
    def set_cap_len(self, max_len):
        self.max_len = max_len
    # function to set the learning rate scheduler
    def set_lr_scheduler(self, scheduler):
        self.scheduler = scheduler  
    # function to set the loss for training
    def set_loss(self, loss):
        self.loss = loss
    def set_dict_loc(self, loc):
        self.dict_loc = loc
    def set_cuda(self):
        self.dtype = torch.cuda.FloatTensor
        self.img_embedder.cuda()
        self.cap_embedder.cuda()
    
    def train_epoch(self, data, batch_size):
        self.img_embedder.train()
        self.cap_embedder.train()
        # for keeping track of the average loss over all batches
        train_loss = 0
        num_batches = 0
        for batch in self.batcher(data, batch_size, shuffle = True):
            # if there is a lr scheduler, take a step in the scheduler
            if self.scheduler:
                self.scheduler.step()
                self.iteration +=1
            # retrieve a minibatch from the batcher
            img, cap, lengths = batch
            num_batches +=1
            # sort the tensors based on the unpadded caption length so they can be used
            # with the pack_padded_sequence function
            cap = cap[np.argsort(- np.array(lengths))]
            img = img[np.argsort(- np.array(lengths))]
            lengths = np.array(lengths)[np.argsort(- np.array(lengths))] 
            
            # convert data to pytorch variables
            img, cap = Variable(self.dtype(img)), Variable(self.dtype(cap))
            # reset the gradients of the optimiser
            self.optimizer.zero_grad()
            # embed the images and audio using the networks
            img_embedding = self.img_embedder(img)
            cap_embedding = self.cap_embedder(cap, lengths)
            # calculate the loss
            loss = self.loss(img_embedding, cap_embedding, cuda = True)
            # calculate the gradients and perform the backprop step
            loss.backward()
            # clip the gradients if required
            #if args.gradient_clipping:
            #    torch.nn.utils.clip_grad_norm(img_net.parameters(), img_clipper.clip)
            #    torch.nn.utils.clip_grad_norm(cap_net.parameters(), cap_clipper.clip)
            # update weights
            self.optimizer.step()
            # add loss to average
            train_loss += loss.data
            print(train_loss.cpu()[0]/num_batches)
        return train_loss/num_batches        

    def test_epoch(self, data, batch_size):
        # set to evaluation mode
        self.img_embedder.eval()
        self.cap_embedder.eval()
        # for keeping track of the average loss
        test_batches = 0
        test_loss = 0
        for batch in self.batcher(data, batch_size, shuffle = False):
            img, cap, lengths = batch
            test_batches += 1
            # sort the tensors based on the unpadded caption length so they can be used
            # with the pack_padded_sequence function
            cap = cap[np.argsort(- np.array(lengths))]
            img = img[np.argsort(- np.array(lengths))]
            lengths = np.array(lengths)[np.argsort(- np.array(lengths))]
     
            # convert data to pytorch variables
            img, cap = Variable(self.dtype(img)), Variable(self.dtype(cap))        
            # embed the images and audio using the networks
            img_embedding = self.img_embedder(img)
            cap_embedding = self.cap_embedder(cap, lengths)
            loss = self.loss(img_embedding, cap_embedding, cuda = True)
            # add loss to average
            test_loss += loss.data 
        return test_loss/test_batches
    # function to save parameters in a results folder
    def save_params(self, epoch, loc):
        torch.save(self.cap_embedder.state_dict(), os.path.join(loc, 'caption_model' + '.' +str(epoch)))
        torch.save(self.img_embedder.state_dict(), os.path.join(loc, 'image_model' + '.' +str(epoch)))
        
class snli_trainer():
    def __init__(self, cap_embedder, classifier, optimizer, loss):
        self.dtype = torch.FloatTensor
        self.long = torch.LongTensor
        self.cap_embedder = cap_embedder
        self.classifier = classifier
        self.optimizer = optimizer
        self.loss = loss
        # set default for the caption length
        self.max_len = 300
    def token_batcher(self, data, batch_size, shuffle):
        return iterate_snli_tokens(data, batch_size, self.dict_loc, self.max_len, shuffle)
    def raw_text_batcher(self, data, batch_size, shuffle):
        return iterate_snli(data, batch_size, self.max_len, shuffle)  
    
    # functions to set which minibatcher to use
    def set_token_batcher(self):
        self.batcher = self.token_batcher
    def set_raw_text_batcher(self):
        self.batcher = self.raw_text_batcher
    # function to set the max frame/word/character length of the captions
    def set_cap_len(self, max_len):
        self.max_len = max_len
    # function to set the learning rate scheduler
    def set_lr_scheduler(self, scheduler):
        self.scheduler = scheduler  
    # function to set the loss for training
    def set_loss(self, loss):
        self.loss = loss
    def set_dict_loc(self, loc):
        self.dict_loc = loc
    def set_cuda(self):
        self.dtype = torch.cuda.FloatTensor
        self.Long = torch.cuda.LongTensor
        self.cap_embedder.cuda()
        self.classifier.cuda()
    def train_epoch(self, data, batch_size):
        global iteration  
        self.cap_embedder.train()
        self.classifier.train()
        train_loss = 0
        num_batches = 0 
        # iterator returns converted sentences and their lenghts and labels in minibatches
        for sen1, len1, sen2, len2, labels in self.batcher(data, batch_size, shuffle = True):
            self.scheduler.step()
            iteration += 1
            num_batches += 1
            # embed the sentences using the encoder.
            sent1 = self.embed(sen1, len1)
            sent2 = self.embed(sen2, len2)
            # predict the class labels using the classifier. 
            prediction = self.classifier(self.feature_vector(sent1, sent2))
            # convert the ground truth text labels of the sentence pairs to indices 
            labels = self.create_labels(labels)
            # calculate the loss and take a training step
            loss = self.loss(prediction, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data
            print(train_loss.cpu()[0]/num_batches)
        return(train_loss.cpu()[0]/num_batches)         

    def test_epoch(self, data, batch_size):
        self.cap_embedder.eval()
        self.classifier.eval()
        # list to hold the scores of each minibatch
        preds = []
        val_loss = 0
        num_batches = 0
        # iterator returns converted sentences and their lenghts and labels in minibatches
        for sen1, len1, sen2, len2, labels in self.batcher(data, batch_size, shuffle = False):
            num_batches += 1
            # embed the sentences using the encoder.
            sent1 = self.embed(sen1, len1)
            sent2 = self.embed(sen2, len2)
            # predict the class label using the classifier. 
            prediction = self.classifier(self.feature_vector(sent1, sent2))
            # convert the ground truth labels of the sentence pairs to indices 
            labels = self.create_labels(labels)
            # calculate the loss
            loss = self.loss(prediction, labels)
            val_loss += loss.data       
            # get the index (i) of the predicted class
            v, i = torch.max(prediction, 1)
            # compare the predicted class to the true label and add the score to the predictions list
            preds.append(torch.Tensor.double(i.cpu().data == labels.cpu().data).numpy())
        # concatenate all the minibatches and calculate the accuracy of the predictions
        preds = np.concatenate(preds)
        # calculate the accuracy based on the number of correct predictions
        correct = np.mean(preds) * 100 
        return val_loss.cpu()[0]/num_batches, correct
    # convenience function to embed a batch of sentences using the network
    def embed(self, sent, length):
        # sort the sentences in descending order by length
        l_sort = np.argsort(- np.array(length))
        sent = sent[l_sort]
        length = np.array(length)[l_sort]   
        # embed the sentences
        sent = Variable(self.dtype(sent))
        emb = self.cap_embedder(sent, length)
        # reverse the sorting again such that the sentence pairs can be properly lined up again    
        emb = emb[self.long(np.argsort(l_sort))]    
        return emb

    # convert the text labels to integers for the sofmax layer.
    def create_labels(self, labels):
        labs = []
        for x in labels:
            if x  == 'contradiction':
                l = 0
            elif x  == 'entailment':
                l = 1
            elif x  == 'neutral':
                l = 2
            # pairs where annotators were indecisive are marked neutral
            else:
                l = 2
            labs.append(l)
        labs = self.dtype(self.long(labs), requires_grad = False)
        return labs

    # create the feature vector for the classifier.
    def feature_vector(self, sent1, sent2):
        # cosine distance
        cosine = torch.matmul(sent1, sent2.t()).diag()
        # absolute elementwise distance
        absolute = (sent1 - sent2).norm(1, dim = 1, keepdim = True)
        # element wise or hadamard product
        elem_wise = sent1 * sent2
        # concatenate the embeddings and the derived features into a single feature vector
        return torch.cat((sent1, sent2, elem_wise, absolute, cosine.unsqueeze(1)), 1)
