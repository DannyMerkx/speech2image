# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import torch
import pickle
sys.path.append('/data/speech2image/PyTorch/functions')
from encoders import text_gru_encoder
from collections import defaultdict

# Set PATHs
PATH_TO_SENTEVAL = '/data/SentEval'
PATH_TO_DATA = '/data/SentEval/data'
glove_loc = '/data/glove.840B.300d.txt'
PATH_TO_ENC = '/data/speech2image/PyTorch/flickr_words/results/caption_model.32'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
dict_loc = '/data/speech2image/preprocessing/dictionaries/combined_dict'

# create a dictionary of all the words in the senteval tasks
all_dictionary = defaultdict(int)

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

def word_2_index(word_list, batch_size, dict_loc):
    global all_dictionary
    w_dict = load_obj(dict_loc)
    # filter words that do not occur in the dictionary
    word_list = [[word if word in w_dict else '<oov>' for word in sent] for sent in word_list]
    max_sent_len = max([len(x) for x in word_list])
    text_batch = np.zeros([batch_size, max_sent_len])
    lengths = []
    for i, words in enumerate(word_list):
        lengths.append(len(words))
        for j, word in enumerate(words):
            text_batch[i][j] = w_dict[word]
            if all_dictionary[word] == 0:
                all_dictionary[word] = len(all_dictionary)
    return text_batch, lengths

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    global the_one_dictionary
    # replace empty captions with the out of vocab token
    batch = [sent if sent != [] else ['<oov>'] for sent in batch]
    # add beginning and end of sentence tokens     
    sents = [['<s>'] + x + ['</s>'] for x in batch]
    embeddings = []
    batchsize = len(sents)
    # turn the captions into indices
    sent, lengths = word_2_index(sents, batchsize, dict_loc)
    sort = np.argsort(- np.array(lengths))    
    sent = sent[sort]
    lengths = np.array(lengths)[sort]
    sent = torch.autograd.Variable(torch.cuda.FloatTensor(sent))    
    # embed the captions
    embeddings = params.sent_embedder(sent, lengths)
    embeddings = embeddings.data.cpu().numpy()    
    embeddings = embeddings[np.argsort(sort)]
    return embeddings

dict_len = len(load_obj(dict_loc))
# create config dictionaries with all the parameters for your encoders
text_config = {'embed':{'num_chars': dict_len, 'embedding_dim': 300, 'sparse': False, 'padding_idx': 0}, 
               'gru':{'input_size': 300, 'hidden_size': 2048, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 4096, 'hidden_size': 128, 'heads': 1}}
# create encoder
encoder = text_gru_encoder(text_config)
for p in encoder.parameters():
    p.requires_grad = False
encoder.cuda()
# load pretrained netowrk
encoder_state = torch.load(PATH_TO_ENC)
encoder.load_state_dict(encoder_state)
# load glove embeddings
encoder.load_embeddings(dict_loc, glove_loc)

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 5, 'epoch_size': 4}
params_senteval['sent_embedder'] = encoder
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      ]
    results = se.eval(transfer_tasks)
    print(results)
    #save_obj(the_one_dictionary, 'the_one_dictionary')

