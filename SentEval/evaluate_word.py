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
from encoders import char_gru_encoder
from collections import defaultdict

# Set PATHs
PATH_TO_SENTEVAL = '/data/SentEval'
PATH_TO_DATA = '/data/SentEval/data'
# PATH_TO_VEC = 'glove/glove.840B.300d.txt'
PATH_TO_ENC = '/data/speech2image/PyTorch/flickr_words/results/caption_model.32'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
#dict_loc = '/data/speech2image/preprocessing/dictionaries/snli_indices'
dict_loc = '/data/speech2image/preprocessing/dictionaries/combined_dict'
the_one_dictionary = defaultdict(int)

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

def word_2_index(word_list, batch_size, dict_loc):
    global the_one_dictionary
    w_dict = load_obj(dict_loc)
    # filter words that do not occur in the dictionary
    word_list = [[word for word in sent if word in w_dict] for sent in word_list]
    # If a sentence has no words occuring in the dictionary replace it with the end of sentence token
    for x in range(len(word_list)):
        if word_list[x] == []:
            word_list[x] = ['</s>']
    max_sent_len = max([len(x) for x in word_list])
    text_batch = np.zeros([batch_size, max_sent_len])
    lengths = []
    for i, words in enumerate(word_list):
        lengths.append(len(words))
        for j, word in enumerate(words):
            text_batch[i][j] = w_dict[word]
            if the_one_dictionary[word] == 0:
                the_one_dictionary[word] = len(the_one_dictionary)
    return text_batch, lengths

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    global the_one_dictionary
    batch = [sent if sent != [] else ['.'] for sent in batch] 
    #sents = [' '.join(s) for s in batch]    
    sents = [['<s>'] + x + ['</s>'] for x in batch]
    #sents = [['<bos>'] + x + ['<eos>'] for x in batch]
    embeddings = []
    batchsize = len(sents)
    sent, lengths = word_2_index(sents, batchsize, dict_loc)
    sort = np.argsort(- np.array(lengths))    
    sent = sent[sort]
    lengths = np.array(lengths)[sort]
    sent = torch.autograd.Variable(torch.cuda.FloatTensor(sent))    

    embeddings = params.sent_embedder(sent, lengths)
    embeddings = embeddings.data.cpu().numpy()    
    embeddings = embeddings[np.argsort(sort)]
    return embeddings

dict_len = len(load_obj(dict_loc))
# create config dictionaries with all the parameters for your encoders
char_config = {'embed':{'num_chars': dict_len, 'embedding_dim': 300, 'sparse': False, 'padding_idx': 0}, 
               'gru':{'input_size': 300, 'hidden_size': 2048, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 4096, 'hidden_size': 128, 'heads': 1}}

encoder = char_gru_encoder(char_config)
encoder.cuda()

glove_loc = '/data/glove.840B.300d.txt'
#encoder.load_embeddings(dict_loc, glove_loc)

encoder_state = torch.load(PATH_TO_ENC)
#encoder.load_state_dict(encoder_state)

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
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      ]
    

    results = se.eval(transfer_tasks)
    print(results)
    #save_obj(the_one_dictionary, 'the_one_dictionary')

