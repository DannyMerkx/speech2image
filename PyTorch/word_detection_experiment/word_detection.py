#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:56:48 2019

@author: danny
"""
import pickle
import tables
import numpy as np
import torch
import sys
import torch.nn as nn
from collections import defaultdict

sys.path.append('/data/speech2image/PyTorch/functions')
from data_split import split_data_flickr
from costum_layers import multi_attention

dict_path = '/data/caption2image/PyTorch/flickr_words/flickr_frequency'
trained_loc = '/data/speech2image/PyTorch/flickr_audio/mfcc_results/caption_model.28'
data_loc = '/prep_data/flickr_features.h5'
split_loc = '/data/flickr/dataset.json'

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

# from a dictionary containing word frequencies, select words which occur over 
# x times, less than y times and are over z characters long
def select(dic, x,y,z):
     l = defaultdict(int)
     for k in dic.keys():
         if dic[k] < y and dic[k] > x and len(k) > z:
             if l[k] == 0:
                 l[k] = len(l)
     return l

def iterate_flickr(h5_file):
    for x in h5_file.root:
        yield x
        
# the 5fold minibatchers are for the datasets with 5 captions per image (mscoco, flickr). It returns all 5 captions per image.
def iterate_audio_5fold(f_nodes, batchsize, shuffle = True):
    max_frames = 2048
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for i in range(0, 5):
        for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
            # take a batch of nodes of the given size               
            excerpt = f_nodes[start_idx:start_idx + batchsize]
            speech = []
            images = []
            caption = []
            lengths = []
            for ex in excerpt:
                # extract and append the vgg16 features
                images.append(eval('ex.' + 'resnet' + '._f_list_nodes()[0].read()'))
                # extract the audio features
                sp = eval('ex.' + 'mfcc' + '._f_list_nodes()[i].read().transpose()')
                # padd to the given output size
                n_frames = sp.shape[1]
		
                if n_frames < max_frames:
                    sp = np.pad(sp, [(0, 0), (0, max_frames - n_frames )], 'constant')
                # truncate to the given input size
                if n_frames > max_frames:
                    sp = sp[:,:max_frames]
                    n_frames = max_frames
                lengths.append(n_frames)
                speech.append(sp)
                # extract the caption
                cap = eval('ex.' + 'tokens' + '._f_list_nodes()[i].read()')
                cap = [x.decode('utf-8') for x in cap]
                caption.append(cap)
            max_length = max(lengths)
            # reshape the features and recast as float64
            speech = np.float64(speech)
            caption = np.array(caption)
            # truncate all padding to the length of the longest utterance
            speech = speech[:,:, :max_length]
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, speech, caption, lengths
            
# load the word frequency dictionary     
f_dict = load_obj(dict_path)
# select words which occur between 50 and a 1000 times and are over 3 characters long
words = select(f_dict, 50, 1000, 3)
vocab_size = len(words) 
# open and load the data
data_file = tables.open_file(data_loc, mode='r+') 
f_nodes = [node for node in iterate_flickr(data_file)]
# split the data
train, val, test = split_data_flickr(f_nodes, split_loc)
################################network config##################################
# rnn encoder for audio (mfcc, mbn etc.)
class audio_rnn_encoder(nn.Module):
    def __init__(self, config):
        super(audio_rnn_encoder, self).__init__()
        conv = config['conv']
        rnn= config['rnn']
        att = config ['att'] 
        self.Conv = nn.Conv1d(in_channels = conv['in_channels'], 
                                  out_channels = conv['out_channels'], kernel_size = conv['kernel_size'],
                                  stride = conv['stride'], padding = conv['padding'])
        self.RNN = nn.GRU(input_size = rnn['input_size'], hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['num_layers'], batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], dropout = rnn['dropout'])
        self.att = multi_attention(in_size = att['in_size'], hidden_size = att['hidden_size'], n_heads = att['heads'])
        
    def forward(self, input, l):
        x = self.Conv(input)
        # update the lengths to compensate for the convolution
        l = [int((y-(self.Conv.kernel_size[0]-self.Conv.stride[0]))/self.Conv.stride[0]) for y in l]
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(2,1), l, batch_first=True)
        x, hx = self.RNN(x)       
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
        return x
    
# this class removes the attention layer from the rnn encoder. Use this to load
# only part of the network e.g. up to the first second or third rnn layer. 
class audio_rnn_sublayers(nn.Module):
    def __init__(self, config):
        super(audio_rnn_sublayers, self).__init__()
        conv = config['conv']
        rnn= config['rnn']
        self.Conv = nn.Conv1d(in_channels = conv['in_channels'], 
                                  out_channels = conv['out_channels'], kernel_size = conv['kernel_size'],
                                  stride = conv['stride'], padding = conv['padding'])
        self.RNN = nn.GRU(input_size = rnn['input_size'], hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['num_layers'], batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], dropout = rnn['dropout'])
        self.pool = nn.functional.avg_pool1d
        
    def forward(self, input, l):
        x = self.Conv(input)
        # update the lengths to compensate for the convolution
        l = [int((y-(self.Conv.kernel_size[0]-self.Conv.stride[0]))/self.Conv.stride[0]) for y in l]
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = torch.nn.utils.rnn.pack_padded_sequence(x.transpose(2,1), l, batch_first=True)
        x, hx = self.RNN(x)      
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)

        x = nn.functional.normalize(self.pool(x.permute(0,2,1), x.size(1)).squeeze(), p=2, dim=1)  
        return x
    
# the word detection network
class det_net(torch.nn.Module):
    def __init__(self, in_features, vocab):
        super(det_net, self).__init__()
        self.linear_transform = torch.nn.Linear(in_features = in_features, out_features = vocab)
        torch.nn.init.xavier_uniform(self.linear_transform.weight.data)
    def forward(self, input):
        x = self.linear_transform(input)
        return x
###############################################################################################
        
# network configurations for the caption embedder. the embedder for the third layer can use the
# full audio config and just skip the attention layer
audio_config = {'conv':{'in_channels': 39, 'out_channels': 64, 'kernel_size': 6, 'stride': 2,
               'padding': 0, 'bias': False}, 'rnn':{'input_size': 64, 'hidden_size': 1024, 
               'num_layers': 3, 'batch_first': True, 'bidirectional': True, 'dropout': 0}, 
               'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}}       

# class to hold the detection networks and training/test functions
class word_detection_network():
    def __init__(self, config, cap_net, trained_loc, n, det_net, vocab_size):
        super(word_detection_network, self).__init__()
        self.vocab_size = vocab_size
        # create the caption embedder
        self.cap_net = cap_net(config)
        # load pretrained parameters and fix the parameters
        cap_state = torch.load(trained_loc)
        while len(cap_state) > n:
            cap_state.popitem(-1)
        self.cap_net.load_state_dict(cap_state, strict = False)
        for param in self.cap_net.parameters():
            param.requires_grad = False
        self.cap_net.eval
        # create the word detection classifier, optimiser and lr scheduler
        self.classifier = det_net(2048, vocab_size)
        self.optimizer = torch.optim.Adam(list(self.classifier.parameters()), 0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size= 1,
                                                 gamma = 1)
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        # train function
    def train(self, word_dict, data):
        self.cap_net.cuda()
        self.classifier.cuda()
        train_loss = 0
        iteration = 0
        for batch in iterate_audio_5fold(data, 32, shuffle = True):
            img, speech, cap, lengths = batch
            # sort the captions by length (required for the rnn)
            speech = speech[np.argsort(- np.array(lengths))]
            cap = cap[np.argsort(- np.array(lengths))]
            lengths = np.array(lengths)[np.argsort(- np.array(lengths))]  
            speech = torch.cuda.FloatTensor(speech)
            # create the training targets using the text captions
            cap_embedding = self.cap_net(speech, lengths)
            targets = torch.zeros(32, self.vocab_size)
            for i, c in enumerate(cap):
                for w in c:
                    if word_dict[w] -1 >= 0:
                        targets[i, word_dict[w]-1] = 1
            # predict from the caption embedding, calculate the loss and update the weights
            preds = self.classifier(cap_embedding)
            loss = self.loss_function(preds, targets.cuda())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.data
            iteration +=1
            if np.mod(iteration, 100) ==0:
                print(train_loss/iteration)
        print(train_loss/iteration)
        self.scheduler.step()
        self.cap_net.cpu()
        self.classifier.cpu()        
        # test function
    def test(self, word_dict, data):
        self.cap_net.cuda()
        self.classifier.cuda()
        n = 21  # create 20 evenly spaced detection thresholds      
        threshold = np.linspace(1e-6,1,n)
        TP = [0 for x in range(n)]
        FP = [0 for x in range(n)]
        FN = [0 for x in range(n)]
        TN = [0 for x in range(n)]
        predictions = torch.zeros(0,0).cuda()
        targets = torch.zeros(0,0)
        for batch in iterate_audio_5fold(data, 32, shuffle = False):
            img, speech, cap, lengths = batch
            # sort the captions by length (required for the rnn)
            speech = speech[np.argsort(- np.array(lengths))]
            cap = cap[np.argsort(- np.array(lengths))]
            lengths = np.array(lengths)[np.argsort(- np.array(lengths))]  
            speech = torch.cuda.FloatTensor(speech)
            # embed the spoken caption
            cap_embedding = self.cap_net(speech, lengths)
            # create the test targets using the text captions
            targs = torch.zeros(32, self.vocab_size)
            for i, c in enumerate(cap):
                for w in c:
                    if words[w] -1 >= 0:
                        targs[i, words[w]-1] = 1
            # predict from the caption embedding and append prediction to the list
            preds = torch.sigmoid(self.classifier(cap_embedding))
            predictions = torch.cat([predictions, preds.data])
            targets = torch.cat([targets, targs.data])
        # for each threshold, check the detected words and calculate true positive/negate and false positive/negative
        for i, thresh in enumerate(threshold):
            thresh_preds = predictions.ge(thresh).float()
            thresh_preds = thresh_preds.cpu()
            TP[i] += torch.sum((thresh_preds + targets) == 2).data.numpy()
            FP[i] += torch.sum(thresh_preds > targets).data.numpy()
            FN[i] += torch.sum(thresh_preds < targets).data.numpy()
            TN[i] += torch.sum((thresh_preds + targets) == 0).data.numpy()
        # print the results for this epoch
        print('True Positives:' + str(TP))
        print('True Negatives:' + str(TN))
        print('False Positives:' + str(FP))
        print('False Negatives:' + str(FN))
    
        self.cap_net.cpu()
        self.classifier.cpu()
        
full_net = word_detection_network(audio_config, audio_rnn_encoder, trained_loc,
                                  30, det_net, vocab_size)
three_layer_net = word_detection_network(audio_config, audio_rnn_sublayers, trained_loc,
                                  26, det_net, vocab_size)
audio_config['rnn']['num_layers'] = 2
two_layer_net = word_detection_network(audio_config, audio_rnn_sublayers, trained_loc,
                                  18, det_net, vocab_size)
audio_config['rnn']['num_layers'] = 1
single_layer_net = word_detection_network(audio_config, audio_rnn_sublayers, trained_loc,
                                  10, det_net, vocab_size)
untrained_net = word_detection_network(audio_config, audio_rnn_sublayers, trained_loc,
                                  2, det_net, vocab_size)

for x in range(1,33):
    #print('full model, epoch' + str(x))
    #full_net.train(words, train)
    #full_net.test(words, val)
    print('untrained model, epoch' + str(x))
    untrained_net.train(words, train)
    untrained_net.test(words, val)
    #print('single layer model, epoch' + str(x))
    #single_layer_net.train(words, train)
    #single_layer_net.test(words, val)
    #print('second layer model, epoch' + str(x))
    #two_layer_net.train(words, train)
    #two_layer_net.test(words, val)
    #print('third layer model, epoch' + str(x))
    #three_layer_net.train(words, train)
    #three_layer_net.test(words, val)

#print('full model, test')
#full_net.test(words, test)
print('untrained model, test')
untrained_net.test(words, test)
#print('single layer model, test')
#single_layer_net.test(words, test)
#print('second layer model, test')
#two_layer_net.test(words, test)
#print('third layer model, test')
#three_layer_net.test(words, test)  
