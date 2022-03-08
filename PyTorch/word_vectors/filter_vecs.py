#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 08:59:49 2021
filter all vectors that you want to test to use only the intersection of 
words that are available in all. This ensures N is the same for all 
evaluations
@author: danny
"""
import numpy as np
import gensim as gs

def load_vecs(file_loc, keys = None):      
    file = open(file_loc)
    if keys:     
        vecs = [x.split() for x in file if x.split(' ')[0] in keys]  
    else:
        vecs = [x.split() for x in file]      
    vecs = {v[0]: np.array([float(x) for x in v[1:]]) for v in vecs}      
    file.close()
    return vecs


def save_vecs(loc, vecs, intersect):
    with open(loc, 'w') as file:
        for key in vecs.keys():
            if key in intersect:
                line = f'{key} {" ".join([str(x) for x in vecs[key]])}'
                file.write(line + '\n')   

i2c_vectors = load_vecs('./word_vectors/unfiltered/my_vecs.txt')

glove_costum = load_vecs('./word_vectors/unfiltered/glove.txt')

fasttext_costum = load_vecs('./word_vectors/unfiltered/fasttext_300_sg.txt')

w2v_costum = load_vecs('./word_vectors/unfiltered/word2vec_300_sg.txt')


glove_pt = '../../../glove.840B.300d.txt' 

fasttext_pt = './word_vectors/fasttext2M.vec'

w2v_pt = './word_vectors/word2vec_news.bin'

glove_vecs = load_vecs(glove_pt, i2c_vectors.keys())
        
ft_vecs = load_vecs(fasttext_pt, i2c_vectors.keys())

w2v_vecs = gs.models.KeyedVectors.load_word2vec_format(w2v_pt, binary = True)
words = w2v_vecs.key_to_index
w2v = {}
for key in words.keys():
    if key in i2c_vectors.keys():
        w2v[key] =  w2v_vecs[key]

intersect = set(i2c_vectors.keys()).intersection(set(glove_costum.keys()))
intersect = intersect.intersection(set(fasttext_costum.keys()))
intersect = intersect.intersection(set(w2v_costum.keys()))
intersect = intersect.intersection(set(glove_vecs.keys()))
intersect = intersect.intersection(set(ft_vecs.keys()))
intersect = intersect.intersection(set(w2v.keys()))

save_vecs('./word_vectors/grounded_vecs.txt', i2c_vectors, intersect)
save_vecs('./word_vectors/glove.txt', glove_costum, intersect)
save_vecs('./word_vectors/fasttext.txt', fasttext_costum, intersect)
save_vecs('./word_vectors/word2vec.txt', w2v_costum, intersect)
save_vecs('./word_vectors/glove_pretrained.txt', glove_vecs, intersect)
save_vecs('./word_vectors/fasttext_pretrained.txt', ft_vecs, intersect)
save_vecs('./word_vectors/word2vec_pretrained.txt', w2v, intersect)
