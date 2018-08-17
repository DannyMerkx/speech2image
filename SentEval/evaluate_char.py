# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

# This function loads a pretrained character model and runs it through the SentEval library.
# To use, download SentEval from github and place this script in the examples folder.

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import torch
import string
sys.path.append('/data/speech2image/PyTorch/functions')
from encoders import char_gru_encoder

# Set PATHs
PATH_TO_SENTEVAL = '/data/SentEval'
PATH_TO_DATA = 'data/SentEval/data'
# PATH_TO_VEC = 'glove/glove.840B.300d.txt'
# path to the pretrained encoder model
PATH_TO_ENC = '/data/speech2image/PyTorch/flickr_char/results/caption_model.20'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def find_index(char):
    # define the set of valid characters.
    valid_chars = string.printable
    return valid_chars.find(char)

# convert characters to indices
def char_2_index(raw_text, batch_size):
    max_sent_len = max([len(x) for x in raw_text])
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

# SentEval prepare and batcher
# prepare is not needed
def prepare(params, samples):
    return
# batcher to run the sentence batches through the encoder
def batcher(params, batch):
    # concatenate the tokens into a sentence. 
    batch = [sent if sent != [] else ['.'] for sent in batch] 
    sents = [' '.join(s) for s in batch]    
    embeddings = []
    batchsize = len(sents)
    # convert the characters to indices and return sentence lenght
    sent, lengths = char_2_index(sents, batchsize)
    # order the batch by length
    sort = np.argsort(- np.array(lengths))    
    sent = sent[sort]
    lengths = np.array(lengths)[sort]
    # make torch variables
    sent = torch.autograd.Variable(torch.cuda.FloatTensor(sent))    
    # embed the sentences
    embeddings = params.sent_embedder(sent, lengths)
    embeddings = embeddings.data.cpu().numpy()
    # reorder them to the original order    
    embeddings = embeddings[np.argsort(sort)]

    return embeddings

# create config dictionaries with all the parameters for your encoders
char_config = {'embed':{'num_chars': 100, 'embedding_dim': 20, 'sparse': False, 'padding_idx': 0}, 
               'gru':{'input_size': 20, 'hidden_size': 1024, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1}}

encoder = char_gru_encoder(char_config)
encoder.cuda()
encoder_state = torch.load(PATH_TO_ENC)
encoder.load_state_dict(encoder_state)

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 10}
params_senteval['sent_embedder'] = encoder
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
