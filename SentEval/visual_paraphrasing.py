#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:00:26 2018

@author: danny
"""

import numpy as np
import torch
import tables
from scipy.io import loadmat
import string
import sys
sys.path.append('/data/speech2image/PyTorch/functions')

from encoders import snli, char_gru_encoder

PATH_TO_ENC = '/data/speech2image/PyTorch/flickr_char/results/caption_model.20'

x = loadmat('/home/danny/Downloads/imagine_v1/data/visual_paraphrasing/dataset.mat')

gt = x['gt']
sents_1 = x['sentences_1']
sents_2 = x['sentences_2']

# convert the data from the nested array structed to a more sensible list of lists.
sents_1 = [[z[0] for z in y] for x in sents_1 for y in x[0]]
sents_2 = [[z[0] for z in y] for x in sents_2 for y in x[0]]

char_config = {'embed':{'num_chars': 100, 'embedding_dim': 20, 'sparse': False, 'padding_idx': 0},
               'gru':{'input_size': 20, 'hidden_size': 1024, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 4}}

in_feats = (char_config['gru']['hidden_size'] * 2**char_config['gru']['bidirectional']) * char_config['att']['heads']* 3 * 3 + 2
classifier_config = {'in_feats': in_feats, 'hidden': 512, 'class': 2}

encoder = char_gru_encoder(char_config)
encoder.cuda()
encoder_state = torch.load(PATH_TO_ENC)
encoder.load_state_dict(encoder_state)

cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index = -100)
cosine_loss = torch.nn.CosineEmbeddingLoss()
classifier = snli(classifier_config)

optimizer = torch.optim.Adam(list(classifier.parameters()), 0.0001)

epoch = 0
n_epochs = 100
batch_size = 100

def find_index(char):
    # define the set of valid characters.
    valid_chars = string.printable
    return valid_chars.find(char)

# convert characters to indices
def char_2_index(raw_text, batch_size, max_sent_len):
    text_batch = np.zeros([batch_size, max_sent_len])
    # keep track of the origin sentence length to use in pack_padded_sequence
    lengths = []
    for i, text in enumerate(raw_text):
        lengths.append(len(text))        
        for j, char in enumerate(text):
            if not find_index(char) == -1:
                text_batch[i][j] = find_index(char)
            else:
                text_batch[i][j] = find_index('')
    return text_batch, lengths


def embed(sent1, l1, sent2, l2, labels, sent_embedder):
    
    # order the batch by length
    sort1 = []
    sort2 = []
    for x in range(len(l1)):
        sort1.append(np.argsort(- np.array(l1[x])) )
        sort2.append(np.argsort(- np.array(l2[x])) )
    for x in range(len(sent1)):        
        sent1[x] = sent1[x][sort1[x]]
        sent2[x] = sent2[x][sort2[x]]
        # make torch variables
        sent1[x] = torch.autograd.Variable(torch.cuda.FloatTensor(sent1[x]))
        sent2[x] = torch.autograd.Variable(torch.cuda.FloatTensor(sent2[x]))
    for x in range(len(l1)):
        l1[x] = np.array(l1[x])[sort1[x]]
        l2[x] = np.array(l2[x])[sort2[x]]
    
    embeddings1 = []
    embeddings2 = []    
    for x in range(len(sent1)):
        embeddings1.append(sent_embedder(sent1[x], l1[x])[0])
        embeddings1[x] = embeddings1[x][torch.cuda.LongTensor(np.argsort(sort1[x]))]
        embeddings2.append(sent_embedder(sent2[x], l2[x])[0])
        embeddings2[x] = embeddings2[x][torch.cuda.LongTensor(np.argsort(sort2[x]))]
    
    return embeddings1, embeddings2

def vp_batcher(sents_1, sents_2, gt, batchsize, shuffle = True):
        n_samples = len(sents_1)
        data = list(zip(sents_1, sents_2, gt))
        if shuffle:
            # optionally shuffle the input
            np.random.shuffle(data)
        for start_idx in range(0, n_samples - batchsize + 1, batchsize):
            excerpt = data[start_idx:start_idx + batchsize]
            sent1, sent2, labels = zip(*excerpt)
            sent1 = list(sent1)
            sent2 = list(sent2)
            # converts the sentences to character ids and sentence lengths
            max_words_1 = max([len(y) for x in sent1 for y in x])
            max_words_2 = max([len(y) for x in sent2 for y in x])
            sent1, l1 = zip(char_2_index([x[0] for x in sent1], batchsize, max_words_1),
                            char_2_index([x[1] for x in sent1], batchsize, max_words_1),
                            char_2_index([x[2] for x in sent1], batchsize, max_words_1))
            sent2, l2 = zip(char_2_index([x[0] for x in sent2], batchsize, max_words_2),
                            char_2_index([x[1] for x in sent2], batchsize, max_words_2),
                            char_2_index([x[2] for x in sent2], batchsize, max_words_2))
            yield list(sent1), list(l1), list(sent2), list(l2), list(labels)

def feature_vector(sent1, sent2):
    sent1 = torch.sum(torch.stack(sent1), 0)
    sent2 = torch.sum(torch.stack(sent2), 0)
    # cosine distance
    cosine = torch.matmul(sent1, sent2.t()).diag()
    # absolute elementwise distance
    absolute = (sent1 - sent2).norm(1, dim = 1, keepdim = True)
    # element wise or hadamard product
    elem_wise = sent1 * sent2
    # concatenate the embeddings and the derived features into a single feature vector
    return torch.cat((sent1, sent2, elem_wise, absolute, cosine.unsqueeze(1)), 1)

def train(sents_1, sents_2, gt, batch_size):         
    batcher = vp_batcher(sents_1, sents_2, gt, batch_size)
    preds = []
    train_loss = 0
    num_batches = 0 
    classifier.train()
    for sent1, l1, sent2, l2, labels in batcher:
        num_batches += 1
        emb_1, emb_2 = embed(sent1, l1, sent2, l2, labels, encoder)
        labels = torch.autograd.Variable(torch.LongTensor([int(x) for x in labels]))
        prediction = classifier(feature_vector(emb_1, emb_2))
        loss = cross_entropy_loss(prediction, labels)
        loss += cosine_loss(torch.sum(torch.stack(emb_1),0), torch.sum(torch.stack(emb_2),0), labels * 2 -1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        print(train_loss.cpu()[0]/num_batches)
        v, i = torch.max(prediction, 1)
        # compare the predicted class to the true label and add the score to the predictions list
        preds.append(torch.Tensor.double(i.cpu().data == labels.cpu().data).numpy())
        # concatenate all the minibatches and calculate the accuracy of the predictions
    preds = np.concatenate(preds)
    # calculate the accuracy based on the number of correct predictions
    correct = np.mean(preds) * 100 
    return(train_loss.cpu()[0]/num_batches), correct
    
while epoch <= n_epochs:
    epoch += 1
    loss, correct = train(sents_1, sents_2, gt, batch_size)
    print('epoch 1 accuracy: ' + str(correct) + ' loss: ' + str(loss))