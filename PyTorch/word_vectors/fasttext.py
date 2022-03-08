#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:09:50 2021

@author: danny
"""
from collections import defaultdict
from gensim.models import FastText

text = []        
with open('./../../../coco_text.txt', 'r') as file:
    text = [line for line in file]
text = [x.replace('\n', '').split(' ') for x in text]    
file.close()

words = defaultdict(int)
for sent in text:
    for word in sent:
        words[word] = 1

# train word2vec on the mscoco data and store the model
model = FastText(sentences=text, vector_size=300, window=10, min_count=1, 
                 workers=4, sg = 1, negative = 10, epochs = 10)

#model.save("fasttext.model")
# extract and save the vectors
out = open('word_vectors/fasttext_300_sg.txt', 'w')
for x in words.keys():
    try:
        vec = model.wv[x]
        formatted = x + ' ' + ' '.join([str(y) for y in vec])
        out.write(formatted + '\n')   
    except:
        continue
out.close()    