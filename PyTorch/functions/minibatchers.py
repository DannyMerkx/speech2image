"""
Created on Wed Feb 14 13:06:06 2018

@author: danny
minibatchers for the neural network training. There are three batchers for 
speech, tokens and raw text. Each batcher has a 5 fold version as many image 
captioning databases have multiple (5) captions per image. The batchers also 
return the lenghts of the captions in the batch so it can be used with torch
pack_padded_sequence.
"""
import numpy as np
import string
import pickle
import torch
import tables
from torch.utils.data import Dataset, Sampler
##############################################################################
# the following functions are used to convert the input strings to indices for 
# the word embedding layers

# loader for the dictionary, loads a pickled dictionary.
def load_obj(loc):
    with open(f'{loc}.pkl', 'rb') as f:
        return pickle.load(f)

# function to find the indices of characters. 
def find_index(char, valid_chars = False):
    # define the set of valid characters, use printable as default
    if valid_chars == False:
        valid_chars = string.printable 
    idx = valid_chars.find(char)
    # add 1 to the index so idx 0 is left free for the padding index
    if idx != -1:
        idx += 1
    return idx

# function to turn the character based text caption into character indices
def char_2_index(batch, batch_size):
    # get the maximum sentence length, the entire batch must be padded to this 
    # length
    max_sent_len = max([len(x) for x in batch])
    index_batch = np.zeros([batch_size, max_sent_len])
    # keep track of the original sentence length to use in pack_padded_sequence
    lengths = []
    for i, text in enumerate(batch):
        lengths.append(len(text))        
        for j, char in enumerate(text):
            index_batch[i][j] = find_index(char)
    return index_batch, lengths

def word_2_index(batch, batch_size, dict_loc):
    w_dict = load_obj(dict_loc)
    # filter out words that do not occur in the dictionary
    batch = [[word if word in w_dict else '<oov>' for word in sent] for sent in batch]
    # get the maximum sentence length, the entire batch must be padded to this 
    # length
    max_sent_len = max([len(x) for x in batch])
    index_batch = np.zeros([batch_size, max_sent_len])
    lengths = []
    # load the indices for the words from the dictionary
    for i, words in enumerate(batch):
        lengths.append(len(words))
        for j, word in enumerate(words):
            index_batch[i][j] = w_dict[word]
    return index_batch, lengths

class PlacesDataset(Dataset):
    def __init__(self, h5_file, visual, audio, transform=None):
        self.f_nodes = [node for node in self.read_data(h5_file)] 
        
        self.train = []
        self.test = []
        for idx, node in enumerate(self.f_nodes):
            if node._f_getattr('train'):
                self.train.append(idx)
            else:
                self.test.append(idx)
        
        self.val = self.train[:1000]
        self.train = self.train[1000:]
        
        self.visual = visual
        self.audio = audio
    def read_data(self, h5_file):
        h5_file = tables.open_file(h5_file, 'r+')
        for x in h5_file.root:
            for y in x._f_list_nodes():
                yield y
    def __len__(self):
        return len(self.f_nodes)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = eval(f'self.f_nodes[idx].{self.visual}._f_list_nodes()[0].read()')
        speech = eval(f'self.f_nodes[idx].{self.audio}._f_list_nodes()[0].read().transpose()')        
        return {'im': image, 'sp': speech}
    
class pad_fn():
    def __init__(self, max_len):
        self.max_len = max_len   
    def pad(self, batch):
        # determine the length of the longest sentence in the batch
        batch_max = max([x['sp'].shape[1] for x in batch])
        # determine if the batch max is longer than the global allowed max
        if batch_max > self.max_len:
            # set the batch max to the globally allowed max
            batch_max = self.max_len
        lengths = []  
        speech = []
        for x in batch:
            sp = x['sp']           
            n_frames = sp.shape[1]
            if n_frames < batch_max:
                sp = np.pad(sp, [(0, 0), (0, batch_max - n_frames )], 'constant')            
            if n_frames > batch_max:
                sp = sp[:,:batch_max]
                n_frames = batch_max  
            lengths.append(n_frames)  
            speech.append(torch.FloatTensor(sp))
        im_batch = torch.stack([torch.FloatTensor(x['im']) for x in batch])
        sp_batch = torch.stack(speech)        
        return im_batch, sp_batch, lengths
    def __call__(self, batch):
        return self.pad(batch)
 
class PlacesSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source, mode = 'train'):
        if mode == 'train':
            self.split = data_source.train
        elif mode == 'val':
            self.split = data_source.val
        else:
            self.split = data_source.test
    def __iter__(self):        
        return iter(self.split)
    def __len__(self):
        return len(self.data_source)

################################### minibatchers ##############################

# minibatcher which takes a list of nodes and returns the visual and audio 
# features, possibly resized. Visual and audio should contain a string of the
# names of the visual and audio features nodes in the h5 file. Frames is the 
# max length of the time sequence, the batcher truncates to this length.
def iterate_audio(f_nodes, batchsize, visual, audio, max_len = 2048, 
                  shuffle = True):  
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):  
        # take a batch of nodes of the given size               
        excerpt = f_nodes[start_idx:start_idx + batchsize]        
        speech = []
        images = []
        lengths = []
        for ex in excerpt:
            # extract and append the visual features
            images.append(eval(f'ex.{visual}._f_list_nodes()[0].read()'))
            # retrieve the audio features
            sp = eval(f'ex.{audio}._f_list_nodes()[0].read().transpose()')
            n_frames = sp.shape[1]
            # pad to max_len if sentence is shorter
            if n_frames < max_len:
                sp = np.pad(sp, [(0, 0), (0, max_len - n_frames )], 'constant')
            # truncate to max_len if sentence is longer
            if n_frames > max_len:
                sp = sp[:,:max_len]
                n_frames = max_len
            lengths.append(n_frames)
            speech.append(sp)
      
        max_batch = max(lengths)
        # reshape the features and recast as float64
        speech = np.float64(speech)
        # truncate all padding to the length of the longest utterance
        speech = speech[:,:, :max_batch]
        # reshape the features into appropriate shape and recast as float32
        images = np.float64(images)
        yield images, speech, lengths  

# batcher for character input. Keeps track of the unpadded senctence lengths 
# to use with pytorch's pack_padded_sequence. Optionally shuffle.
def iterate_char(f_nodes, batchsize, visual, text, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size               
        excerpt = f_nodes[start_idx:start_idx + batchsize]
        caption = []
        images = []
        for ex in excerpt:
            # extract and append the visual features
            images.append(eval(f'ex.{visual}._f_list_nodes()[0].read()'))
            # extract the audio features
            cap = eval(f'ex.{text}._f_list_nodes()[i].read()')
            cap = cap.decode('utf-8')
            # append an otherwise unused character as a start of sentence 
            # character and convert the sentence to lower case.
            caption.append(cap)
        # converts the sentence to character ids. 
        caption, lengths = char_2_index(caption, batchsize)
        images = np.float64(images)
        yield images, caption, lengths

# batcher for token input. Keeps track of the unpadded senctence lengths to 
# use with pytorch's pack_padded_sequence. Requires a pre-defined dictionary 
# mapping the tokens to indices. Optionally shuffle.
def iterate_tokens(f_nodes, batchsize, visual, text, dict_loc, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size               
        excerpt = f_nodes[start_idx:start_idx + batchsize]
        caption = []
        images = []
        for ex in excerpt:
            # extract and append the visual features
            images.append(eval(f'ex.{visual}._f_list_nodes()[0].read()'))
            # extract the audio features
            cap = eval(f'ex.{text}._f_list_nodes()[i].read()')
            cap = [x.decode('utf-8') for x in cap]
            # append an otherwise unused character as a start of sentence 
            # character and convert the sentence to lower case.
            caption.append(cap)
        # converts the sentence to character ids. 
        caption, lengths = word_2_index(caption, batchsize, dict_loc)
        images = np.float64(images)
        yield images, caption, lengths


# batcher for audio input. Keeps track of the unpadded senctence lengths to 
# use with pytorch's pack_padded_sequence. Optionally shuffle.
def iterate_audio_5fold(f_nodes, batchsize, visual, audio, max_len = 2048, 
                        shuffle = True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for i in range(0, 5):
        for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
            # take a batch of nodes of the given size               
            excerpt = f_nodes[start_idx:start_idx + batchsize]
            speech = []
            images = []
            lengths = []
            for ex in excerpt:
                # extract and append the visual features
                images.append(eval(f'ex.{visual}._f_list_nodes()[0].read()'))
                # extract the audio features
                sp = eval(f'ex.{audio}._f_list_nodes()[i].read().transpose()')
                n_frames = sp.shape[1]
                # pad if sentence is shorter than max_len
                if n_frames < max_len:
                    sp = np.pad(sp, [(0, 0), (0, max_len - n_frames )], 
                                'constant'
                                )
                # truncate if sentence is longer than max_len
                if n_frames > max_len:
                    sp = sp[:,:max_len]
                    n_frames = max_len
                lengths.append(n_frames)
                speech.append(sp)
            
            max_batch = max(lengths)
            # reshape the features and recast as float64
            speech = np.float64(speech)
            # if no sentence reaches max_len, the batch can be truncated to the
            # max len in the batch for efficiency 
            speech = speech[:,:, :max_batch]
            images = np.float64(images)
            yield images, speech, lengths

# batcher for character input. Keeps track of the unpadded senctence lengths 
# to use with pytorch's pack_padded_sequence. Optionally shuffle.
def iterate_char_5fold(f_nodes, batchsize, visual, text, max_len = 256, 
                       shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for i in range(0,5):
        for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
            # take a batch of nodes of the given size               
            excerpt = f_nodes[start_idx:start_idx + batchsize]
            caption = []
            images = []
            for ex in excerpt:
                # extract and append the visual features
                images.append(eval(f'ex.{visual}._f_list_nodes()[0].read()'))
                # extract the audio features and truncate to max_len
                cap = eval(f'ex.{text}._f_list_nodes()[i].read()')
                cap = cap.decode('utf-8')[:max_len]
                # append an otherwise unused character as a start of sentence 
                # character and 
                # convert the sentence to lower case.
                caption.append(cap)
            # converts the sentence to character indices. 
            caption, lengths = char_2_index(caption, batchsize)
            images = np.float64(images)
            yield images, caption, lengths

# batcher for token input. Keeps track of the unpadded senctence lengths to use 
# with pytorch's pack_padded_sequence. Requires a pre-defined dictionary 
# mapping the tokens to indices. Optionally shuffle.
def iterate_tokens_5fold(f_nodes, batchsize, visual, text, dict_loc, 
                         max_len = 64, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for i in range(0,5):
        for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
            # take a batch of nodes of the given size               
            excerpt = f_nodes[start_idx:start_idx + batchsize]
            caption = []
            images = []
            for ex in excerpt:
                # extract and append the visual features
                images.append(eval(f'ex.{visual}._f_list_nodes()[0].read()'))
                # extract the audio features
                cap = eval(f'ex.{text}._f_list_nodes()[i].read()')
                # add begin of sentence and end of sentence tokens and truncate
                # to max_len
                cap = ['<s>'] + [x.decode('utf-8') for x in cap[:max_len]] \
                      + ['</s>']                            
                caption.append(cap)
            # converts the sentence to word indices. 
            caption, lengths = word_2_index(caption, batchsize, dict_loc)
            images = np.float64(images)
            yield images, caption, lengths
