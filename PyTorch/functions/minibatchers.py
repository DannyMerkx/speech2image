"""
Created on Wed Feb 14 13:06:06 2018

@author: danny
minibatchers for the neural network training. Recently refactored to use torch
dataset structures for reusability of the components. Typically you need
a dataset which loads and splits your data, a collate function which determines
how data is combined into minibatches and a sampler which determines the data 
order
"""
import string
import pickle
import tables
import json

import numpy as np

from torch.utils.data import Dataset, Sampler
from collections import defaultdict
######################### utility functions ###################################
# the following functions are used to convert the input strings to indices for 
# the embedding layers

# loader for the dictionary, loads a pickled dictionary.
def load_obj(loc):
    with open(f'{loc}.pkl', 'rb') as f:
        return pickle.load(f)

# class to turn the character based text caption into character indices
class char_2_index():
    def __init__(self, valid_chars = False):
        if valid_chars == False:
            self.valid_chars = string.printable
        else:
            self.valid_chars = valid_chars            
    def find_index(self, sent):
        index_sent = np.zeros(len(sent))
        for i, char in enumerate(sent):
            idx = self.valid_chars.find(char)
            # add one to valid indices to leave 0 free as the padding index
            if idx != -1:
                idx += 1
            index_sent[i] = idx 
        return index_sent
# class to turn word based text captions into indices
class word_2_index():
    def __init__(self, dict_loc):    
        self.w_dict = load_obj(dict_loc)
    def find_index(self, sent):
        # filter out words that do not occur in the dictionary
        sent = [word if word in self.w_dict else '<oov>' for word in sent]
        index_sent = np.zeros(len(sent))
        for i, word in enumerate(sent):
            index_sent[i] = self.w_dict[word]
        return index_sent
    
########################### Datasets #########################################
class PlacesDataset(Dataset):
    def __init__(self, h5_file, visual, cap, split_files, transform=None):        
        self.f_nodes = [node for node in self.read_data(h5_file)]

        self.train, self.val, self.val_unseen, self.test, self.test_unseen = \
        self.split_data_places(split_files)
        
        self.visual = visual
        self.cap = cap
    def read_data(self, h5_file):       
        h5_file = tables.open_file(h5_file, 'r')
        for subgroup in h5_file.root:
            for node in subgroup:
                yield node
    def __len__(self):
        return len(self.val) + len(self.train) + len(self.test) + \
               len(self.val_unseen) + len(self.test_unseen)
    def __getitem__(self, node):
        image = eval(f'node.{self.visual}._f_list_nodes()[0].read()')
        caption = eval(f'node.{self.cap}._f_list_nodes()[0].read().transpose()')        
        return {'im': image, 'cap': caption}
    def split_data_places(self, split_files):
        split_dict = defaultdict(str)
        for loc in split_files.keys():
            file = open(loc, 'r')
            file = json.load(file)     
            for f in file['data']: 
                split_dict[f['uttid'].replace('-', '_')] = split_files[loc]           
        train = []
        val = []
        val_unseen = []
        test = []
        test_unseen = []
        oov = []
        for idx, node in enumerate(self.f_nodes):
            name = node._v_name.replace('places_', '')
            if split_dict[name] == 'train':
                train.append(node)
            elif split_dict[name] == 'dev':
                val.append(node)    
            elif split_dict[name] == 'test':
                test.append(node) 
            elif split_dict[name] == 'dev_unseen':
                val_unseen.append(node) 
            elif split_dict[name] == 'test_unseen':
                test_unseen.append(node) 
            else:
                oov.append(node)
            if oov:
                print(f'could not find split for {len(oov)} files')
        return train, val, val_unseen, test, test_unseen

class FlickrDataset(Dataset):
    def __init__(self, h5_file, visual, cap, split_loc, transform=None):
        # get all the nodes in the h5 file
        self.f_nodes = [node for node in self.read_data(h5_file)]
        # split into sets using the json split file
        self.train, self.val, self.test = self.split_data_flickr(split_loc)
        self.visual = visual
        self.cap = cap
    def read_data(self, h5_file):
        h5_file = tables.open_file(h5_file, 'r')
        for x in h5_file.root:
                yield x
    def __len__(self):
        return len(self.f_nodes)
    def __getitem__(self, sample):
        idx, node = sample
        # node should be a tuple of feature node and its caption idx (1-5) 
        image = eval(f'node.{self.visual}._f_list_nodes()[0].read()')
        caption = eval(f'node.{self.cap}._f_list_nodes()[idx].read().transpose()')        
        return {'im': image, 'cap': caption} 
    def split_data_flickr(self, loc):
        file = json.load(open(loc))
        split_dict = {}
        for x in file['images']:
            split_dict[x['filename'].replace('.jpg', '')] = x['split']       
        train = []
        val = []
        test = []   
        for idx, node in enumerate(self.f_nodes):
            name = node._v_name.replace('flickr_', '')
            if split_dict[name] == 'train':
                train.append(node)
            elif split_dict[name] == 'val':
                val.append(node)    
            elif split_dict[name] == 'test':
                test.append(node) 
        return train, val, test

############################# Batch collate functions #########################

# receives a batch of audio and images, reshaping the audio to either given 
# max len or the batch max len (whichever is shortest) 
class audio_pad_fn():
    def __init__(self, max_len, dtype):
        self.dtype = dtype
        self.max_len = max_len   
    def pad(self, batch):
        # determine the length of the longest sentence in the batch
        batch_max = max([x['cap'].shape[1] for x in batch])
        # determine if the batch max is longer than the global allowed max
        if batch_max > self.max_len:
            # set the batch max to the globally allowed max
            batch_max = self.max_len
        lengths = []  
        cap = []
        for x in batch:
            # pad or truncate the caption according to the batch max length
            c = x['cap']           
            n_frames = c.shape[1]
            if n_frames < batch_max:
                c = np.pad(c, [(0, 0), (0, batch_max - n_frames )], 'constant')            
            if n_frames > batch_max:
                c = c[:,:batch_max]
                n_frames = batch_max  
            lengths.append(n_frames)  
            cap.append(c)
        # convert to proper torch datatype
        im_batch = self.dtype(np.array([x['im'] for x in batch]))
        cap_batch = self.dtype(np.array(cap))        
        return im_batch, cap_batch, lengths
    def __call__(self, batch):
        return self.pad(batch)

# receives a batch of tex captions and images, reshaping the caption to either
# given max len or the batch max len (whichever is shortest) 
class token_pad_fn():
    def __init__(self, max_len, dtype, token_type = 'char'):
        self.dtype = dtype
        self.max_len = max_len 
        self.token_type = token_type
        if token_type == 'char':
            self.token_2_idx = char_2_index()
        elif token_type == 'word':
            self.token_2_idx = word_2_index()
            # subtract 2 from maxlen to account for sos and eos tokens
            self.max_len -= 2
        else:
            print('invalid token type')
    # decoded the byte encoded text captions
    def decode(self, sent):
        if self.token_type == 'char':
            return sent.decode('utf-8')
        else:
            return [w.decode('utf-8') for w in sent]
    
    def pad(self, batch):
        # determine the length of the longest sentence in the batch
        batch_max = max([x['cap'].shape[1] for x in batch])
        # determine if the batch max is longer than the global allowed max
        if batch_max > self.max_len:
            # set the batch max to the globally allowed max
            batch_max = self.max_len
        lengths = []  
        cap = []
        for x in batch:
            # decode input and pad or truncate the caption according to the
            # batch max length
            c = self.decode(x['cap'])
            n_tokens = len(c)
            if n_tokens < batch_max:
                c = np.pad(c, [0, batch_max - n_tokens], 'constant')            
            if n_tokens > batch_max:
                c = c[:batch_max]
                n_tokens = batch_max
            if self.token_type == 'word':
                c = ['<s>'] + c + ['</s>']
                n_tokens += 2            
            lengths.append(n_tokens) 
            # convert the input to character indices
            c = self.token_2_idx.find_index(c)
            cap.append(c)
        # convert to proper torch datatype
        im_batch = self.dtype(np.array([x['im'] for x in batch]))
        cap_batch = self.dtype(np.array(cap))        
        return im_batch, cap_batch, lengths
    
    def __call__(self, batch):
        return self.pad(batch)

################################ Batch samplers ###############################
class PlacesSampler(Sampler):
    def __init__(self, data_source, mode = 'train', shuffle = False):
        if mode == 'train':
            self.split = list(data_source.train)
            if shuffle:
                np.random.shuffle(self.split)
        elif mode == 'val':
            self.split = list(data_source.val)
        else:
            self.split = list(data_source.test)
    def __iter__(self):        
        return iter(self.split)
    def __len__(self):
        return len(self.data_source)

class FlickrSampler(Sampler):
    def __init__(self, data_source, mode = 'train', shuffle = False):
        if mode == 'train':
            self.split = list(data_source.train)
            if shuffle:
                np.random.shuffle(self.split)
        elif mode == 'val':
            self.split = list(data_source.val)
        else:
            self.split = list(data_source.test)
    def __iter__(self):     
        # the iterator pairs ints 1-5 to each of the node indexes to make sure
        # all 5 captions per image are used but no 2 captions of the same img
        # ever end up in the same batch
        return iter([(idx, node) for idx in range(5) for node in self.split])
    def __len__(self):
        return len(self.data_source)    
    
    

class ParaphrasingDataset(Dataset):
    def __init__(self, h5_file, cap, split_loc, transform=None):
        # get all the nodes in the h5 file
        self.f_nodes = [node for node in self.read_data(h5_file)]
        # split into sets using the json split file
        self.train, self.val, self.test = self.split_data_flickr(split_loc)
        self.cap = cap
    def read_data(self, h5_file):
        h5_file = tables.open_file(h5_file, 'r')
        for x in h5_file.root:
                yield x
    def __len__(self):
        return len(self.f_nodes)
    def __getitem__(self, sample):
        idx, node = sample
        # node should be a tuple of feature node and its caption idx (1-5) 
        cap_1 = eval(f'node.{self.cap}._f_list_nodes()[idx[0]].read().transpose()')
        cap_2 = eval(f'node.{self.cap}._f_list_nodes()[idx[1]].read().transpose()')        
        return {'cap_1': cap_1, 'cap_2': cap_2} 
    def split_data_flickr(self, loc):
        file = json.load(open(loc))
        split_dict = {}
        for x in file['images']:
            split_dict[x['filename'].replace('.jpg', '')] = x['split']       
        train = []
        val = []
        test = []   
        for idx, node in enumerate(self.f_nodes):
            name = node._v_name.replace('flickr_', '')
            if split_dict[name] == 'train':
                train.append(node)
            elif split_dict[name] == 'val':
                val.append(node)    
            elif split_dict[name] == 'test':
                test.append(node) 
        return train, val, test

class ParaphrasingSampler(Sampler):
    def __init__(self, data_source, mode = 'train', shuffle = False):
        if mode == 'train':
            self.split = list(data_source.train)
            if shuffle:
                np.random.shuffle(self.split)
        elif mode == 'val':
            self.split = list(data_source.val)
        else:
            self.split = list(data_source.test)
    def __iter__(self):     
        com = [(1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), 
               (4,5)]
        # the iterator pairs ints 1-5 to each of the node indexes to make sure
        # all 5 captions per image are used but no 2 captions of the same img
        # ever end up in the same batch
        return iter([(idx, node) for idx in com for node in self.split])
    def __len__(self):
        return len(self.data_source)    

# receives a batch of audio and images, reshaping the audio to either given 
# max len or the batch max len (whichever is shortest) 
class audio_pad_para():
    def __init__(self, max_len, dtype):
        self.dtype = dtype
        self.max_len = max_len   
    def pad(self, batch):
        # determine the length of the longest sentence in the batch
        batch_max1 = max([x['cap_1'].shape[1] for x in batch])
        batch_max2 = max([x['cap_2'].shape[1] for x in batch])
        # determine if the batch max is longer than the global allowed max
        if batch_max1 > self.max_len:
            # set the batch max to the globally allowed max
            batch_max1 = self.max_len
        if batch_max2 > self.max_len:
            batch_max2 = self.max_len
        lengths = []
        lengths_2 = []
        cap = []
        cap_2 = []
        for x in batch:
            # pad or truncate the caption according to the batch max length
            c = x['cap_1']
            n_frames = c.shape[1]
            if n_frames < batch_max1:
                c = np.pad(c, [(0, 0), (0, batch_max1 - n_frames )], 'constant')            
            if n_frames > batch_max1:
                c = c[:,:batch_max1]
                n_frames = batch_max1  
            lengths.append(n_frames)  
            cap.append(c)
            
        for x in batch:
            # pad or truncate the caption according to the batch max length
            c = x['cap_2']
            n_frames = c.shape[1]
            if n_frames < batch_max2:
                c = np.pad(c, [(0, 0), (0, batch_max2 - n_frames )], 'constant')            
            if n_frames > batch_max2:
                c = c[:,:batch_max2]
                n_frames = batch_max2 
            lengths_2.append(n_frames)  
            cap_2.append(c)
            
        # convert to proper torch datatype
        cap_batch = self.dtype(np.array(cap))   
        cap_batch2 = self.dtype(np.array(cap))
        return cap_batch, cap_batch2, lengths, lengths_2
    def __call__(self, batch):
        return self.pad(batch)