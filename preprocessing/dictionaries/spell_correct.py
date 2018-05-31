#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:39:34 2018

@author: danny
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 15:55:39 2018

@author: danny
"""
import numpy as np
import string
from nltk.corpus import wordnet
from nltk.corpus import stopwords
# list all single character edits to a word
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)



def create_spell_check_dict(coco_dict):
    stop_words = set(stopwords.words('english')) 
        # open a dictionary file
    with open('/data/speech2image/preproccesing/dictionaries/large.txt') as file:
        x = file.read()
    
    # split the dictionary add punctuation as valid 'words'
    spell = x.split()
    for x in string.punctuation:
        spell.append(x)
    # make a dictionary to speed up search
    spelling_dict = {}
    for x in spell:
        spelling_dict[x] = x
        
    # get all the words that appear in mscoco that do not match any dictionary or wordnet word
    misspelled = {}
    for x in coco_dict.keys():
        if wordnet.synsets(x) == [] and not x in stop_words and not x in spelling_dict:
            # add the single edits that would result in a valid dictionary word
            misspelled[x] = [x for x in list(edits1(x)) if not wordnet.synsets(x) == [] or x in stop_words or x in spelling_dict]
    
    # remove spelling suggestions for words containing digits 
    pop = []
    for x in misspelled.keys():
        for y in string.digits:
            if y in x:
                pop.append(x)
    # pop the digits from the misspelled list
    pop = list(set(pop))
    for x in pop:
        misspelled.pop(x)  

    # remove spelling suggestions for words with 's as these are not valid spellings but do not occur in 
    # wordnet or the dictionary
    pop = []
    for x in misspelled.keys():
        if '\'s' in x:        
            pop.append(x)
    # pop the digits from the misspelled list
    pop = list(set(pop))
    for x in pop:
        misspelled.pop(x)              
    
    # a list of all the misspells with only one option for correction        
    single_opt = {}
    
    for x in misspelled.keys():
        if len(misspelled[x]) == 1:
            single_opt[x] = misspelled[x][0]
    # for the ones with multiple options we need a way to choose the best option
    multi_opt = {}
    
    for x in misspelled.keys():
        if len(misspelled[x]) > 1:
            multi_opt[x] = misspelled[x]
    
    # make a list with the probability of the suggested corrections in the current corpus
    count = []
    
    for x in coco_dict.keys():
        count.append(coco_dict[x])
    p = {}
    
    for x in multi_opt.keys():
        p[x] = []
        for y  in multi_opt[x]:
            if y in coco_dict:
                p[x].append(coco_dict[y]/sum(count))
            else:
                p[x].append(0)
    
    # pop the spelling suggestions that do not occur in the corpus            
    pop = []
    for x in p.keys():
        if sum(p[x]) == 0:
            pop.append(x)
    pop = list(set(pop))
    for x in pop:
        multi_opt.pop(x)  
        p.pop(x)
    
    # add the most likely correction to the correction dictionary
    for x in multi_opt.keys():
        most_likely = np.argmax(p[x])
        multi_opt[x] = multi_opt[x][most_likely]
    
    dictionary = {**single_opt, **multi_opt}

    return(dictionary)
