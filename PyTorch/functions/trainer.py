#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:45:31 2018

@author: danny
"""
from minibatchers import iterate_tokens_5fold, iterate_raw_text_5fold, iterate_audio_5fold, iterate_snli_tokens, iterate_snli
from grad_tracker import gradient_clipping
from evaluate import evaluate

import numpy as np
from torch.autograd import Variable
import torch
import os
import time

# trainer for the flickr database. 
class flickr_trainer():
    def __init__(self, img_embedder, cap_embedder, vis, cap):
        # default datatype, change to cuda by calling set_cuda
        self.dtype = torch.FloatTensor
        # set the embedders. Set an empty scheduler to keep this optional.
        self.img_embedder = img_embedder
        self.cap_embedder = cap_embedder
        self.scheduler = []
        # set gradient clipping to false by default
        self.grad_clipping = False
        # names of the features to be loaded by the batcher
        self.vis = vis
        self.cap = cap
        # keep track of an iteration for lr scheduling
        self.iteration = 0
        # keep track of the number of training epochs
        self.epoch = 1
    # the possible minibatcher for all different types of data for the flickr database
    def token_batcher(self, data, batch_size, shuffle):
        return iterate_tokens_5fold(data, batch_size, self.vis, self.cap, self.dict_loc, shuffle)
    def audio_batcher(self, data, batch_size, shuffle):
        return iterate_audio_5fold(data, batch_size, self.vis, self.cap, shuffle)
    def raw_text_batcher(self, data, batch_size, shuffle):
        return iterate_raw_text_5fold(data, batch_size, self.vis, self.cap, shuffle)    

################## functions to set class values and attributes ###############
    # functions to set which minibatcher to use. Needs to be called as no default is set.
    def set_token_batcher(self):
        self.batcher = self.token_batcher
    def set_raw_text_batcher(self):
        self.batcher = self.raw_text_batcher
    def set_audio_batcher(self):
        self.batcher = self.audio_batcher
    # function to set the learning rate scheduler
    def set_lr_scheduler(self, scheduler):
        self.scheduler = scheduler  
    # function to set the loss for training. Loss is not necessary e.g. when you 
    # only want to test a pretrained model.
    def set_loss(self, loss):
        self.loss = loss
    # set an optimizer. Optional like the loss in case of using just pretrained models.
    def set_optimizer(self, optim):
        self.optimizer = optim
    # set a dictionary for models trained on tokens
    def set_dict_loc(self, loc):
        self.dict_loc = loc
    # set data type and the networks to cuda
    def set_cuda(self):
        self.dtype = torch.cuda.FloatTensor
        self.img_embedder.cuda()
        self.cap_embedder.cuda()
    # manually set the epoch to some number e.g. if continuing training from a 
    # pretrained model
    def set_epoch(self, epoch):
        self.epoch = epoch
    # functions to set new embedders
    def set_img_embedder(self, emb):
        self.img_embedder = emb
    def set_cap_embedder(self, emb):
        self.cap_embedder = emb
    # functions to load a pretrained embedder
    def load_cap_embedder(self, loc):
        cap_state = torch.load(loc)
        self.cap_embedder.load_state_dict(cap_state)
    def load_img_embedder(self, loc):
        img_state = torch.load(loc)
        self.img_embedder.load_state_dict(img_state)
    # optionally load glove embeddings for token based embedders with load_embeddings
    # function implemented.
    def load_glove_embeddings(self, glove_loc):
        self.cap_embedder.load_embeddings(self.dict_loc, glove_loc)
       
################## functions to perform training and testing ##################
    def train_epoch(self, data, batch_size):
        print('training epoch: ' + str(self.epoch))
        # keep track of runtime
        self.start_time = time.time()
        self.img_embedder.train()
        self.cap_embedder.train()
        # for keeping track of the average loss over all batches
        self.train_loss = 0
        num_batches = 0
        for batch in self.batcher(data, batch_size, shuffle = True):
            # if there is a lr scheduler, take a step in the scheduler
            if self.scheduler:
                self.scheduler.step()
                self.iteration +=1
            # retrieve a minibatch from the batcher
            img, cap, lengths = batch
            num_batches +=1
            # embed the images and audio using the networks
            img_embedding, cap_embedding = self.emb(img, cap, lengths)
            # calculate the loss
            loss = self.loss(img_embedding, cap_embedding, cuda = True)
            # reset the gradients of the optimiser
            self.optimizer.zero_grad()
            # calculate the gradients and perform the backprop step
            loss.backward()
            # clip the gradients if required
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm(self.img_embedder.parameters(), self.img_clipper.clip)
                torch.nn.utils.clip_grad_norm(self.cap_embedder.parameters(), self.cap_clipper.clip)
            # update weights
            self.optimizer.step()
            # add loss to average
            self.train_loss += loss.data
            # print loss every n batches
            if num_batches%100 == 0:
                print(self.train_loss.cpu()[0]/num_batches)
        self.train_loss = self.train_loss.cpu()[0]/num_batches
        self.epoch += 1   
    
    def test_epoch(self, data, batch_size):
        # set to evaluation mode
        self.img_embedder.eval()
        self.cap_embedder.eval()
        # for keeping track of the average loss
        test_batches = 0
        self.test_loss = 0
        for batch in self.batcher(data, batch_size, shuffle = False):
            img, cap, lengths = batch
            test_batches += 1      
            # embed the images and audio using the networks
            img_embedding, cap_embedding = self.emb(img, cap, lengths)
            loss = self.loss(img_embedding, cap_embedding, cuda = True)
            # add loss to average
            self.test_loss += loss.data 
        self.test_loss = self.test_loss/test_batches
    
    # embed a batch of images and captions
    def embed(self, img, cap, lengths):
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
        return img_embedding, cap_embedding
######################## evaluation functions #################################
    # report on the time this epoch took and the train and test loss
    def report(self, max_epochs):
        # report on the time and train and val loss for the epoch
        print("Epoch {} of {} took {:.3f}s".format(
                self.epoch, max_epochs, time.time() - self.start_time))
        self.print_train_loss()
        self.print_validation_loss()
    # print the loss values
    def print_train_loss(self):  
        print("training loss:\t\t{:.6f}".format(self.train_loss))
    def print_test_loss(self):        
        print("test loss:\t\t{:.6f}".format(self.test_loss))
    def print_validation_loss(self):
        print("validation loss:\t\t{:.6f}".format(self.test_loss))
    # create and manipulate an evaluator object   
    def set_evaluator(self, n):
        self.evaluator = evaluate(self.dtype, self.img_embedder, self.cap_embedder)
        self.evaluator.set_n(n)
    # calculate the recall@n. Arguments are a set of nodes and a prepend string 
    # (e.g. to print validation or test in front of the results)
    def recall_at_n(self, data, batch_size, prepend):        
        iterator = self.batcher(data, batch_size, shuffle = False)
        # the calc_recall function calculates and prints the recall.
        self.evaluator.embed_data(iterator)
        self.evaluator.print_caption2image(prepend, self.epoch)
        self.evaluator.print_image2caption(prepend, self.epoch)        
    # function to save parameters in a results folder
    def save_params(self, loc):
        torch.save(self.cap_embedder.state_dict(), os.path.join(loc, 'caption_model' + '.' +str(self.epoch)))
        torch.save(self.img_embedder.state_dict(), os.path.join(loc, 'image_model' + '.' +str(self.epoch)))

############ functions to deal with the trainer's gradient clipper ############
    # create a gradient tracker/clipper
    def set_gradient_clipping(self, img_clip_value, cap_clip_value):
        self.grad_clipping = True
        self.img_clipper = gradient_clipping(img_clip_value)
        self.cap_clipper = gradient_clipping(cap_clip_value)  
        self.img_clipper.register_hook(self.img_embedder)
        self.cap_clipper.register_hook(self.cap_embedder)
    # save the gradients collected so far 
    def save_gradients(self, loc):
        self.cap_clipper.save_grads(loc, 'cap_grads')
        self.img_clipper.save_grads(loc, 'img_grads')
    # reset the grads for a new epoch
    def reset_grads(self):
        self.cap_clipper.reset_gradients()
        self.img_clipper.reset_gradients()
    # update the clip value of the gradient clipper based on the previous epoch. Don't call after resetting
    # the grads to 0
    def update_clip(self):
        self.cap_clipper.update_clip_value()
        self.img_clipper.update_clip_value()

    
class snli_trainer():
    def __init__(self, cap_embedder, classifier):
        self.dtype = torch.FloatTensor
        self.long = torch.LongTensor
        self.cap_embedder = cap_embedder
        self.classifier = classifier
        self.scheduler = []
        # set gradient clipping to false by default
        self.grad_clipping = False
        # keep track of an iteration for lr scheduling
        self.iteration = 0
        # keep track of the number of training epochs
        self.epoch = 1
        
    def token_batcher(self, data, batch_size, shuffle):
        return iterate_snli_tokens(data, batch_size, self.dict_loc, shuffle)
    def raw_text_batcher(self, data, batch_size, shuffle):
        return iterate_snli(data, batch_size, shuffle)  

################## functions to set class values and attributes ###############
    # function to set the loss for training. Loss is not necessary e.g. when you 
    # only want to test a pretrained model.
    def set_loss(self, loss):
        self.loss = loss
    # set an optimizer. Optional like the loss in case of using just pretrained models.
    def set_optimizer(self, optim):
        self.optimizer = optim
    # set a dictionary for models trained on tokens
    def set_dict_loc(self, loc):
        self.dict_loc = loc    
    # functions to set which minibatcher to use
    def set_token_batcher(self):
        self.batcher = self.token_batcher
    def set_raw_text_batcher(self):
        self.batcher = self.raw_text_batcher
    # function to set the learning rate scheduler
    def set_lr_scheduler(self, scheduler):
        self.scheduler = scheduler  
    def set_cuda(self):
        self.dtype = torch.cuda.FloatTensor
        self.long = torch.cuda.LongTensor
        self.cap_embedder.cuda()
        self.classifier.cuda()
    # manually set the epoch to some number e.g. if continuing training from a 
    # pretrained model
    def set_epoch(self, epoch):
        self.epoch = epoch
    # functions to set new embedders
    def set_classifier(self, clas):
        self.classifier = clas
    def set_cap_embedder(self, emb):
        self.cap_embedder = emb
    # functions to load a pretrained embedder
    def load_classifier(self, loc):
        clas_state = torch.load(loc)
        self.classifier.load_state_dict(clas_state)
    def load_cap_embedder(self, loc):
        cap_state = torch.load(loc)
        self.cap_embedder.load_state_dict(cap_state)
    # optionally load glove embeddings for token based embedders with load_embeddings
    # function implemented.
    def load_glove_embeddings(self, glove_loc):
        self.cap_embedder.load_embeddings(self.dict_loc, glove_loc)

################## functions to perform training and testing ##################        
    def train_epoch(self, data, batch_size):
        print('training epoch: ' + str(self.epoch))
        # keep track of runtime
        self.start_time = time.time()
        # set the networks to training mode
        self.cap_embedder.train()
        self.classifier.train()
        self.train_loss = 0
        num_batches = 0 
        # iterator returns converted sentences and their lenghts and labels in minibatches
        for sen1, len1, sen2, len2, labels in self.batcher(data, batch_size, shuffle = True):
            # if there is a lr scheduler, take a step in the scheduler
            if self.scheduler:
                self.scheduler.step()
                self.iteration +=1
            
            num_batches += 1
            # embed the sentences using the encoder.
            sent1 = self.embed(sen1, len1)
            sent2 = self.embed(sen2, len2)
            # predict the class labels using the classifier. 
            prediction = self.classifier(self.feature_vector(sent1, sent2))
            # convert the ground truth text labels of the sentence pairs to indices for the softmax layer
            labels = self.create_labels(labels)
            # calculate the loss
            loss = self.loss(prediction, labels)
            # reset the gradients of the optimizer
            self.optimizer.zero_grad()
            # calculate the gradients and perform the backprop step
            loss.backward()
            # clip the gradients if required
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm(self.img_embedder.parameters(), self.img_clipper.clip)
                torch.nn.utils.clip_grad_norm(self.cap_embedder.parameters(), self.cap_clipper.clip)
            self.optimizer.step()
            self.train_loss += loss.data
            if num_batches%1000 == 0:
                print(self.train_loss.cpu()[0]/num_batches)
        self.train_loss = self.train_loss.cpu()[0] / num_batches
        self.epoch += 1

    def test_epoch(self, data, batch_size):
        # set the networks to evaluation mode
        self.cap_embedder.eval()
        self.classifier.eval()
        # list to hold the predictions of each minibatch
        preds = []
        self.test_loss = 0
        num_batches = 0
        # iterator returns converted sentences and their lenghts and labels in minibatches
        for sen1, len1, sen2, len2, labels in self.batcher(data, batch_size, shuffle = False):
            num_batches += 1
            # embed the sentences using the encoder.
            sent1 = self.embed(sen1, len1)
            sent2 = self.embed(sen2, len2)
            # predict the class label using the classifier. 
            prediction = self.classifier(self.feature_vector(sent1, sent2))
            # convert the ground truth labels of the sentence pairs to indices for the softmax layer
            labels = self.create_labels(labels)
            # calculate the loss
            loss = self.loss(prediction, labels)
            self.test_loss += loss.data       
            # get the index (i) of the predicted class
            v, i = torch.max(prediction, 1)
            # compare the predicted class to the true label (i.e. 1 if the prediction is equal to 
            # the ground truth 0 otherwise)
            preds.append(torch.Tensor.double(i.cpu().data == labels.cpu().data).numpy())
        # concatenate all the minibatches and calculate the accuracy of the predictions
        preds = np.concatenate(preds)
        # calculate the accuracy based on the number of correct predictions
        self.accuracy = np.mean(preds) * 100 
        self.test_loss = self.test_loss.cpu()[0] / num_batches
    
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
        labs = Variable(self.long(labs), requires_grad = False)
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
    
############################### evaluation functions ##########################
    # report on the time this epoch took and the train and test loss
    def report(self, max_epochs):
        # report on the time and train and val loss for the epoch
        print("Epoch {} of {} took {:.3f}s".format(
                self.epoch, max_epochs, time.time() - self.start_time))
        self.print_train_loss()
        self.print_validation_loss()
        self.print_accuracy()
    # print the loss values
    def print_train_loss(self):  
        print("training loss:\t\t{:.6f}".format(self.train_loss))
    def print_test_loss(self):        
        print("test loss:\t\t{:.6f}".format(self.test_loss))
    def print_validation_loss(self):
        print("validation loss:\t\t{:.6f}".format(self.test_loss))
    def print_accuracy(self):
        print("validation accuracy:\t\t{:.6f}".format(self.accuracy))
    # function to save parameters in a results folder
    def save_params(self, loc):
        torch.save(self.cap_embedder.state_dict(), os.path.join(loc, 'caption_model' + '.' +str(self.epoch)))
        torch.save(self.classifier.state_dict(), os.path.join(loc, 'classifier' + '.' +str(self.epoch)))

############ functions to deal with the trainer's gradient clipper ############
    # create a gradient tracker/clipper
    def set_gradient_clipping(self, cap_clip_value, class_clip_value):
        self.grad_clipping = True
        self.class_clipper = gradient_clipping(class_clip_value)
        self.cap_clipper = gradient_clipping(cap_clip_value)  
        self.class_clipper.register_hook(self.classifier)
        self.cap_clipper.register_hook(self.cap_embedder)
    # save the gradients collected so far 
    def save_gradients(self, loc):
        self.cap_clipper.save_grads(loc, 'cap_grads')
        self.class_clipper.save_grads(loc, 'classifier_grads')
    # reset the grads for a new epoch
    def reset_grads(self):
        self.cap_clipper.reset_gradients()
        self.class_clipper.reset_gradients()
    # update the clip value of the gradient clipper based on the previous epoch. Don't call after resetting
    # the grads to 0
    def update_clip(self):
        self.cap_clipper.update_clip_value()
        self.class_clipper.update_clip_value()
