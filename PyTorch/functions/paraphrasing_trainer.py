"""
Created on Tue Aug  7 15:45:31 2018
Trainer classes for caption-image retrieval, so that all DNN parts 
can be combined in one trainer object. 
@author: danny
"""
from minibatchers import (FlickrSampler, PlacesSampler, audio_pad_fn,
                          token_pad_fn, audio_pad_para, ParaphrasingSampler)
from grad_tracker import gradient_clipping
from evaluate import evaluate
from torch.utils.data import DataLoader

import torch
import os
import time
# trainer for caption2image models. Combines all DNN parts (optimiser, 
# lr_scheduler, image and caption encoder, batcher, gradient clipper, 
# loss function), training and test loop functions, evaluation functions in one 
# object. 
class flickr_trainer():
    def __init__(self, cap_embedder, cap):
        # default datatype, change if using cuda by calling set_cuda
        self.dtype = torch.FloatTensor
        self.cap_embedder = cap_embedder
        # lr scheduler, grad clipping and attention loss are optional 
        self.scheduler = False
        self.grad_clipping = False
        self.att_loss = False
        # names of the features to be loaded by the batcher
        self.cap = cap
        # keep track of iteration nr for lr scheduling
        self.iteration = 0
        # keep track of the number of training epochs
        self.epoch = 1
    # different feature types require different collate functions for the 
    # batches. Flickr requires a different sampler because it has 5 captions 
    # to an image.
    def audio_batcher(self, data, batch_size, max_len, mode, shuffle):
        return DataLoader(data, batch_size = batch_size, 
                          collate_fn = audio_pad_fn(max_len, self.dtype),
                          sampler = FlickrSampler(data, mode, shuffle))
    def places_batcher(self, data, batch_size, max_len, mode, shuffle):
        return DataLoader(data, batch_size = batch_size, 
                          collate_fn = audio_pad_fn(max_len, self.dtype), 
                          sampler = PlacesSampler(data, mode, shuffle))
    def paraphrasing_batcher(self, data, batch_size, max_len, mode, shuffle):
        return DataLoader(data, batch_size = batch_size, 
                          collate_fn = audio_pad_para(max_len, self.dtype),
                          sampler = ParaphrasingSampler(data, mode, shuffle))
        
############ functions to set the class values and attributes ################
    # minibatcher type should match your input features, no default is set.
    def set_audio_batcher(self):
        self.batcher = self.audio_batcher
    def set_places_batcher(self):
        self.batcher = self.places_batcher
    def set_paraphrasing_batcher(self):
        self.batcher = self.paraphrasing_batcher
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
    def set_cap_embedder(self, emb):
        self.cap_embedder = emb
    # functions to load pretrained models
    def load_cap_embedder(self, loc, device = 'gpu'):
        if device == 'gpu':
            cap_state = torch.load(loc)
        else:
            cap_state = torch.load(loc, map_location = torch.device('cpu'))
        self.cap_embedder.load_state_dict(cap_state)
    # Load glove embeddings for token based embedders, optional. encoder needs
    # to have a load_embeddings method
    def load_glove_embeddings(self, glove_loc):
        self.cap_embedder.load_embeddings(self.dict_loc, glove_loc)
    # functions to enable/disable gradients on the networks, useful to speed
    # up testing
    def no_grads(self):
        for param in self.cap_embedder.parameters():
            param.requires_grad = False
    def req_grads(self):
        for param in self.cap_embedder.parameters():
            param.requires_grad = True

################## functions to perform training and testing ##################
    # training loop
    def train_epoch(self, data, batch_size, n = 100):
        print('training epoch: ' + str(self.epoch))
        # keep track of runtime
        self.start_time = time.time()
        # set networks to training mode
        self.img_embedder.train()
        self.cap_embedder.train()
        # keep track of the average loss over all batches
        self.train_loss = 0
        num_batches = 0
        for batch in self.batcher(data, batch_size, self.cap_embedder.max_len, 
                                  'train', shuffle = True):
            num_batches += 1
            # retrieve a minibatch from the batcher
            cap, cap2, lengths, lengths2 = batch           
            # embed the images and audio using the networks
            cap_embedding, cap_embedding2 = self.embed(cap, cap2, lengths, 
                                                       lengths2)
            # calculate the loss
            loss = self.loss(cap_embedding, cap_embedding2, self.dtype)
            # add vq_loss if the cap_embedder has VQ layers
            if hasattr(self.cap_embedder, 'VQ_loss'):
                loss += self.cap_embedder.VQ_loss
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
            # print loss every n batches
            if num_batches % n == 0:
                print(self.train_loss.cpu().data.numpy()/num_batches)
            # cyclic and step schedulers require updating after every minibatch
            if self.scheduler == 'cyclic' or self.scheduler == 'step':
                self.lr_scheduler.step()
                self.iteration +=1      
        self.train_loss = self.train_loss.cpu().data.numpy()/num_batches
    # test epoch
    def test_epoch(self, data, batch_size, mode):
        # set to evaluation mode to disable dropout
        self.img_embedder.eval()
        self.cap_embedder.eval()
        # set buffers to store the embeddings
        self.image_embeddings = self.dtype()
        self.caption_embeddings = self.dtype()
        # keeping track of the average loss
        test_batches = 0
        self.test_loss = 0
        for batch in self.batcher(data, batch_size, self.cap_embedder.max_len,
                                  mode, shuffle = False):
            # retrieve a minibatch from the batcher
            img, cap, lengths = batch
            test_batches += 1      
            # embed the images and audio using the networks
            img_embedding, cap_embedding = self.embed(img, cap, lengths)
            # store the embeddings in the buffers for later use (R@N)
            self.caption_embeddings = torch.cat((self.caption_embeddings, 
                                                 cap_embedding.data
                                                 )
                                                )
            self.image_embeddings = torch.cat((self.image_embeddings, 
                                               img_embedding.data
                                               )
                                              )                        
            # calculate the loss
            loss = self.loss(img_embedding, cap_embedding, self.dtype)
            # add VQ loss if the cap_embedder has VQ layers
            if hasattr(self.cap_embedder, 'VQ_loss'):
                loss += self.cap_embedder.VQ_loss
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
    def embed(self, cap, cap2, lengths, lengths2):
        # convert data to the right pytorch tensor type
        #img, cap = self.dtype(img), self.dtype(cap)  
        # set requires_grad to false to speed up test/validation epochs
        if not self.cap_embedder.training:
            cap.requires_grad_(False)
            cap2.requires_grad_(False)
        # embed the images and audio using the networks
        cap_embedding = self.cap_embedder(cap, lengths)
        cap_embedding2 = self.cap_embedder(cap2, lengths2)
        return cap_embedding, cap_embedding2
    
######################## evaluation functions #################################
    # report time and evaluation of this training epoch
    def report_training(self, max_epochs, val = False):
        # run a test epoch on the validation set (if available)
        if val != False:
            self.test_epoch(val, 100, 'val')
        t =  time.time() - self.start_time
        print(f'Epoch {self.epoch} of {max_epochs} took {t}s')
        print(f'Training loss: {self.train_loss:.6f}')
        if val != False:
            print(f'Validation loss: {self.test_loss:.6f}')
            # calculate the recall@n on the validation set
            self.evaluator.caption_embeddings = self.caption_embeddings
            self.evaluator.image_embeddings = self.image_embeddings
            self.recall_at_n(val, prepend = 'validation', mode = 'val') 
    
    def report_test(self, test):
        # run a test epoch on the test set
        self.test_epoch(test, 100, 'test')
        print(f'Test loss: {self.test_loss:.6f}')
        # calculate the recall@n on the test set
        self.evaluator.caption_embeddings = self.caption_embeddings
        self.evaluator.image_embeddings = self.image_embeddings
        self.recall_at_n(test, prepend = 'test', mode = 'test')

    # create an evaluator object
    def set_evaluator(self, n):
        self.evaluator = evaluate(self.dtype, self.img_embedder, 
                                  self.cap_embedder
                                  )
        self.evaluator.set_n(n)
    # calculate the recall@n. Arguments are a set of nodes and a prepend string 
    # (e.g. to print validation or test in front of the results). emb = True
    # reembeds the given data
    def recall_at_n(self, data, prepend, mode, emb = False):
        # if you ran a test epoch on the data before calculating recall, the 
        # trainer has buffered the embeddings
        if emb:
            iterator = self.batcher(data, 5, self.cap_embedder.max_len, mode,
                                    shuffle = False)
            # the calc_recall function calculates and prints the recall.
            self.evaluator.embed_data(iterator)
        self.evaluator.print_caption2image(prepend, self.epoch)
        self.evaluator.print_image2caption(prepend, self.epoch)
    def fivefold_recall_at_n(self, prepend):
        # calculates the average recall@n over 5 test folds (for mscoco). 
        # asumes regular r@n on full test set has already been calculated 
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

