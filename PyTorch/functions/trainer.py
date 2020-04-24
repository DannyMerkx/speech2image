#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:45:31 2018
Trainer classes for caption-image retrieval, so that all DNN parts 
can be combined in one trainer object. 
@author: danny
"""
from minibatchers import iterate_tokens_5fold, iterate_char_5fold, iterate_audio_5fold
from grad_tracker import gradient_clipping
from evaluate import evaluate

import torch
import os
import time

# trainer for caption2image models. Combines all DNN parts (optimiser, 
# lr_scheduler, image and caption encoder, batcher, gradient clipper, 
# loss function), training and test loop functions, evaluation functions in one 
# object. 
class flickr_trainer():
    def __init__(self, img_embedder, cap_embedder, vis, cap):
        # default datatype, change if using cuda by calling set_cuda
        self.dtype = torch.FloatTensor
        self.img_embedder = img_embedder
        self.cap_embedder = cap_embedder
        # lr scheduler, grad clipping and attention loss are optional 
        self.scheduler = False
        self.grad_clipping = False
        self.att_loss = False
        # names of the features to be loaded by the batcher
        self.vis = vis
        self.cap = cap
        # keep track of iteration nr for lr scheduling
        self.iteration = 0
        # keep track of the number of training epochs
        self.epoch = 1
    # possible minibatcher types. Fkickr uses 5fold batchers since there are
    # 5 captions per image
    def token_batcher(self, data, batch_size, max_len, shuffle):
        return iterate_tokens_5fold(data, batch_size, self.vis, self.cap, 
                                    self.dict_loc, max_len, shuffle)
    def audio_batcher(self, data, batch_size, max_len, shuffle):
        return iterate_audio_5fold(data, batch_size, self.vis, self.cap, 
                                   max_len, shuffle)
    def raw_text_batcher(self, data, batch_size, max_len, shuffle):
        return iterate_char_5fold(data, batch_size, self.vis, self.cap, 
                                  max_len, shuffle)    

############ functions to set the class values and attributes ################
    # minibatcher type should match your input features, no default is set.
    def set_token_batcher(self):
        self.batcher = self.token_batcher
    def set_raw_text_batcher(self):
        self.batcher = self.raw_text_batcher
    def set_audio_batcher(self):
        self.batcher = self.audio_batcher
    # lr scheduler and attention loss are optional
    def set_lr_scheduler(self, scheduler, s_type):
        self.lr_scheduler = scheduler  
        self.scheduler = s_type
    def set_att_loss(self, att_loss):
        self.att_loss = att_loss
    # loss function and optimizer are required unless testing a pretrained 
    # model
    def set_loss(self, loss):
        self.loss = loss    
    def set_optimizer(self, optim):
        self.optimizer = optim
    # set the dictionary with embedding indices (token based models only)
    def set_dict_loc(self, loc):
        self.dict_loc = loc
    # set data type and the networks to use the gpu if required.
    def set_cuda(self):
        self.dtype = torch.cuda.FloatTensor
        self.img_embedder.cuda()
        self.cap_embedder.cuda()
    # manually set the epoch to some number e.g. if continuing training from a 
    # pretrained model
    def set_epoch(self, epoch):
        self.epoch = epoch
    def update_epoch(self):
        self.epoch += 1
    # functions to replace embedders
    def set_img_embedder(self, emb):
        self.img_embedder = emb
    def set_cap_embedder(self, emb):
        self.cap_embedder = emb
    # functions to load pretrained models
    def load_cap_embedder(self, loc):
        cap_state = torch.load(loc)
        self.cap_embedder.load_state_dict(cap_state)
    def load_img_embedder(self, loc):
        img_state = torch.load(loc)
        self.img_embedder.load_state_dict(img_state)
    # Load glove embeddings for token based embedders, optional. encoder needs
    # to have a load_embeddings method
    def load_glove_embeddings(self, glove_loc):
        self.cap_embedder.load_embeddings(self.dict_loc, glove_loc)
    # functions to enable/disable gradients on the networks, useful to speed
    # up testing
    def no_grads(self):
        for param in self.cap_embedder.parameters():
            param.requires_grad = False
        for param in self.img_embedder.parameters():
            param.requires_grad = False
    def req_grads(self):
        for param in self.cap_embedder.parameters():
            param.requires_grad = True
        for param in self.img_embedder.parameters():
            param.requires_grad = True

################## functions to perform training and testing ##################
    # training loop
    def train_epoch(self, data, batch_size):
        print('training epoch: ' + str(self.epoch))
        # keep track of the runtime
        self.start_time = time.time()
        self.img_embedder.train()
        self.cap_embedder.train()
        # for keeping track of the average loss over all batches
        self.train_loss = 0
        num_batches = 0
        for batch in self.batcher(data, batch_size, self.cap_embedder.max_len, 
                                  shuffle = True):
            # retrieve a minibatch from the batcher
            img, cap, lengths = batch
            num_batches += 1
            # embed the images and audio using the networks
            img_embedding, cap_embedding, dist = self.embed(img, cap, lengths)
            # calculate the loss
            loss = self.loss(img_embedding, cap_embedding, self.dtype)
            loss = loss + torch.abs(dist)/100
            # optionally calculate the attention loss for multihead attention
            if self.att_loss:
                loss += self.att_loss(self.cap_embedder.att, cap_embedding)
            # reset the gradients of the optimiser
            self.optimizer.zero_grad()
            # calculate the gradients and perform the backprop step
            loss.backward()
            # optionally clip the gradients
            if self.grad_clipping:
                torch.nn.utils.clip_grad_norm(self.img_embedder.parameters(), 
                                              self.img_clipper.clip)
                torch.nn.utils.clip_grad_norm(self.cap_embedder.parameters(), 
                                              self.cap_clipper.clip)
            # update the network weights
            self.optimizer.step()
            # add loss to the running average
            self.train_loss += loss.data
            # optionally print loss every n batches
            if num_batches%100 == 0:
                print(self.train_loss.cpu().data.numpy()/num_batches)
            # cyclic and step schedulers require updating after every minibatch
            if self.scheduler == 'cyclic' or self.scheduler == 'step':
                self.lr_scheduler.step()
                self.iteration +=1
                
        self.train_loss = self.train_loss.cpu().data.numpy()/num_batches
    # test epoch
    def test_epoch(self, data, batch_size):
        # set to evaluation mode to disable dropout
        self.img_embedder.eval()
        self.cap_embedder.eval()
        # keeping track of the average loss
        test_batches = 0
        self.test_loss = 0
        for batch in self.batcher(data, batch_size, self.cap_embedder.max_len,
                                  shuffle = False):
            # retrieve a minibatch from the batcher
            img, cap, lengths = batch
            test_batches += 1      
            # embed the images and audio using the networks
            img_embedding, cap_embedding, dist = self.embed(img, cap, lengths)
            # calculate the loss
            loss = self.loss(img_embedding, cap_embedding, self.dtype)
            loss = loss + torch.abs(dist)/100
            # optionally calculate the attention loss for multihead attention
            if self.att_loss:
                loss += self.att_loss(self.cap_embedder.att, cap_embedding)
            # add loss to the running average
            self.test_loss += loss.data 
        self.test_loss = self.test_loss.cpu().data.numpy()/test_batches
        # The 'plateau' scheduler updates the lr if the validation loss 
        # stagnates.                 
        if self.scheduler == 'plateau':
            self.lr_scheduler.step(self.test_loss)   
 
    # Function which combines embeddings the images and captions
    def embed(self, img, cap, lengths):
        # cap = cap[np.argsort(- np.array(lengths))]
        # img = img[np.argsort(- np.array(lengths))]
        # lengths = np.array(lengths)[np.argsort(- np.array(lengths))]   
        
        # convert data to the right pytorch tensor type
        img, cap = self.dtype(img), self.dtype(cap)      
        # embed the images and audio using the networks
        img_embedding = self.img_embedder(img)
        cap_embedding, dist = self.cap_embedder(cap, lengths)
        return img_embedding, cap_embedding, dist
######################## evaluation functions #################################
    # report on the time this epoch took and the train and test loss
    def report(self, max_epochs):
        # report on the time and train and val loss for the epoch
        t =  time.time() - self.start_time
        print(f'Epoch {self.epoch} of {max_epochs} took {t}s')
        self.print_train_loss()
        self.print_validation_loss()
    # print the loss values
    def print_train_loss(self):  
        print(f'Training loss: {self.train_loss:.6f}')
    def print_test_loss(self):        
        print(f'Test loss: {self.test_loss:.6f}')
    def print_validation_loss(self):
        print(f'Validation loss: {self.test_loss:.6f}')
    # create and manipulate an evaluator object   
    def set_evaluator(self, n):
        self.evaluator = evaluate(self.dtype, self.img_embedder, 
                                  self.cap_embedder
                                  )
        self.evaluator.set_n(n)
    # calculate the recall@n. Arguments are a set of nodes and a prepend string 
    # (e.g. to print validation or test in front of the results)
    def recall_at_n(self, data, batch_size, prepend):        
        iterator = self.batcher(data, 5, self.cap_embedder.max_len, 
                                shuffle = False)
        # the calc_recall function calculates and prints the recall.
        self.evaluator.embed_data(iterator)
        self.evaluator.print_caption2image(prepend, self.epoch)
        self.evaluator.print_image2caption(prepend, self.epoch)
    def fivefold_recall_at_n(self, prepend):
        # calculates the average recall@n over 5 folds (for mscoco). 
        self.evaluator.fivefold_c2i('1k ' + prepend, self.epoch)
        self.evaluator.fivefold_i2c('1k ' + prepend, self.epoch)
    # function to save parameters in a results folder
    def save_params(self, loc):
        torch.save(self.cap_embedder.state_dict(), 
                   os.path.join(loc, f'caption_model.{str(self.epoch)}')
                   )
        torch.save(self.img_embedder.state_dict(), 
                   os.path.join(loc, f'image_model.{str(self.epoch)}')
                   )

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
    # update the clip value of the gradient clipper based on the previous epoch
    # Don't call after resetting the grads to 0
    def update_clip(self):
        self.cap_clipper.update_clip_value()
        self.img_clipper.update_clip_value()

