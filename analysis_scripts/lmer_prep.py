#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:07:20 2021
prepare the recognition experiment data for LMER analysis in Julia
@author: danny
"""
import json
import spacy
import tables
import nltk

import numpy as np
import pandas as pd
from collections import defaultdict
from patn_test import evaluator

# load a spacy model
nlp = spacy.load('en_core_web_lg')
# open the flickr metadata
loc = '/home/danny/Documents/databases/Flickr8k/dataset.json'

data = json.load(open(loc))
# process the caption transcripts using spacy
ims = [y['raw'] for x in data['images'] for y in x['sentences'] if x['split'] == 'train']
proc = list(nlp.pipe(ims))
################################## frequency count ##########################
# load the test utterances
data_loc = './test_features.h5'
data = tables.open_file(data_loc, mode='a')
utterances = data.root.audio._f_list_nodes()
# get all unique target words in the test set
word_count = defaultdict(int)
for utt in utterances:
    word_count[utt.transcript.read().decode()]
# count the occurence of each target word in the test set
for sent in proc:
    for w in sent:
        if w.text.lower() in list(word_count.keys()):
            word_count[w.text.lower()] += 1

############################### neighbourhood density #########################
def edit_distance(pron, phon):
    # create all possible deletions, insertions and replacements
    dels = [pron[:x] + pron[x+1:] for x in range(len(pron))]
    ins = [pron[:x] + [ph] + pron[x:] for x in range(len(pron)+1) for ph in phon]
    reps = [pron[:x] + [ph] + pron[x+1:] for x in range(len(pron)) for ph in phon]
    edits = []
    # remove doubles
    for x in dels + ins + reps:
        if x not in edits:
            edits.append(x)
    return edits

# get all possible tokens with only alphabetical chars in flickr
flickr_vocab = list(set([word.norm_ for sent in proc for word in sent if word.is_alpha]))
# get all target words
target_words = list(word_count.keys())

# map target words to their lemma, so we don't consider words with the same
# lemma as phonetic neighbour
word_2_lemma = defaultdict(int)
for utt in utterances:
    word_2_lemma[utt.transcript.read().decode()]
for sent in proc:
    for w in sent:
        lemma = w.lemma_.lower()
        if w.text.lower() in list(word_2_lemma.keys()):
            word_2_lemma[w.text.lower()] = lemma
            
# correct several lemmas
word_2_lemma['oceans'] = 'ocean'
word_2_lemma['parks'] = 'park'
word_2_lemma['pools'] = 'pool'
word_2_lemma['beaches'] = 'beach'
word_2_lemma['swinging'] = 'swing'
word_2_lemma['sunglasses'] = 'sunglasses'
word_2_lemma['shorts'] = 'shorts'
word_2_lemma['splashing'] = 'splash'
word_2_lemma['lay'] = 'lay'
word_2_lemma['laying'] = 'lay'
word_2_lemma['racing'] = 'race'
word_2_lemma['swimming'] = 'swim'
# import the cmu dictionary
cmu = nltk.corpus.cmudict.dict()

prons = {}
missing = []
for w in list(set(flickr_vocab + target_words)):
    # surfs and skateboards aren't in the cmu dict.
    if w == 'surfs':
        ph = [['S', 'ER1', 'F', 'S']]
    elif w == 'skateboards':
        ph = [['S', 'K', 'EY1', 'T', 'B', 'AO2', 'R', 'D', 'Z']]
    else:
        try:
            ph = cmu[w.lower()]
        except:
            # non-target words that aren't in the cmu dict are not problematic
            # but keep track of them anyways
            missing.append(w)
            continue
    # remove the stress markers from the phones
    prons[w] = [[x[:-1] if len(x) > 2 else x for x in pr] for pr in ph]

# create a list of all unique phones in the pronounciations
phones = list({phone: 0 for word in prons.keys() for pron in prons[word] for phone in pron}.keys())
    
# create the neighbourhood density for all target words. Make sure all target words
# have an entry in the pronounciation dictionary
n_density = defaultdict(int)
for word in word_2_lemma.keys():
    # get all possible single edits to the current target word
    edits = []
    for pron in prons[word]:
        edits.extend(edit_distance(pron, phones))
    # possible neighbours exclude words with the same lemma
    poss_neighbours = [pr for w in prons.keys() for pr in prons[w] if (w not in word_2_lemma.keys()) or (word_2_lemma[w] != word_2_lemma[word])] 
    # density is then the overlap between edits and possible neighbours
    dens = 0
    for e in edits:
        if e in poss_neighbours:
            dens += 1
    
    n_density[word] = dens

init_cohort = defaultdict(list)
for word in word_2_lemma.keys():
    pron = prons[word][0]
    
    for x in range(len(pron)):
        n_cohort = 0
        gate = pron[:x+1]
        for key in prons.keys():
            if (gate == prons[key][0][:x+1]) and ((key not in word_2_lemma.keys()) or (word_2_lemma[key] != word_2_lemma[word])):
                n_cohort += 1
        init_cohort[word].append(n_cohort)

############################# prepare dataframe ###############################

# get lemma counts
lemma_count = defaultdict(int)
lemma = {}
for utt in utterances:
    lemma_count[utt.lemma.read().decode()]
    lemma[utt.transcript.read().decode()] = utt.lemma.read().decode()
for w in word_count:
    lemma_count[word_2_lemma[w]] += word_count[w]


lemmas = {x:lemma_count[word_2_lemma[x]] for x in word_count.keys()}
lemmas = pd.DataFrame.from_dict(lemmas, orient = 'index', columns = ['lemma_count'])

# create a dataframe with all the necessary information
df = pd.DataFrame.from_dict(word_count, orient = 'index', columns = ['word_count'])
df['lemma_count'] = lemmas
lemma = pd.DataFrame.from_dict(lemma, orient = 'index', columns = ['lemma'])
df['word'] = df.index
df= df.join(lemma)
prons = {}
for w in list(target_words):
    # surfs and skateboards aren't in the cmu dict.
    if w == 'surfs':
        ph = [['S', 'ER1', 'F', 'S']]
    elif w == 'skateboards':
        ph = [['S', 'K', 'EY1', 'T', 'B', 'AO2', 'R', 'D', 'Z']]
    else:
        ph = cmu[w.lower()]
    # remove the stress markers from the phones
    prons[w] = '_'.join(ph[0])
    
df['phones'] = pd.DataFrame.from_dict(prons, orient = 'index', columns = ['phones'])
df['n_phones'] = [len(x.split('_')) for x in df['phones']]
df['n_vowels'] = [len([x for x in y.split('_') if x[-1] in '0123']) for y in df['phones']]
df['n_cons'] = df['n_phones'] - df['n_vowels']

########################## get word recognition results #######################
def dict_to_df(dictionary):
    temp = pd.Series()
    for x in dictionary.keys():
        for y in dictionary[x].keys():
            
            temp = pd.concat([temp, pd.Series(dictionary[x][y], index = [y[:-2]])])
    return temp
########################### subject 1 ########################################
data_loc = './test_features.h5'
# word types can be tested seperately just in case
word_types = ['singular', 'plural', 'root', 'third', 'participle']   

cap_model = '../models/caption_model.16'
img_model = '../models/image_model.16'

patn_evaluator = evaluator(f_loc = data_loc, img_loc = img_model, 
                           cap_loc = cap_model, enc_type = 'rnn')

patn_evaluator.recognition(word_types, no_gating = True, n = 10, s_id = 's1')
recognition = patn_evaluator.conv_rec_dict()
durations = patn_evaluator.durations

cap_model = '../models/caption_model.16VQ'
img_model = '../models/image_model.16VQ'

patn_evaluator_VQ = evaluator(f_loc = data_loc, img_loc = img_model, 
                              cap_loc = cap_model, enc_type = 'rnn_VQ')

patn_evaluator_VQ.recognition(word_types, no_gating = True, n = 10, s_id = 's1')
recognition_VQ = patn_evaluator_VQ.conv_rec_dict()
durations_VQ = patn_evaluator_VQ.durations

# turn the recognition results into dataframes
p_10 = dict_to_df(recognition)
dur = dict_to_df(durations)
p_10vq = dict_to_df(recognition_VQ)

s1 = pd.concat([p_10, dur, p_10vq],1)
s1.columns = ['p@10', 'dur', 'p@10VQ']
s1['s_id'] = 's1'
s1['word_type'] = ['singular']*49 + ['plural']*42 + ['root']*49 +['third']*49 + ['participle']*49

################################# subject 2 ###################################

cap_model = '../models/caption_model.16'
img_model = '../models/image_model.16'

patn_evaluator = evaluator(f_loc = data_loc, img_loc = img_model, 
                           cap_loc = cap_model, enc_type = 'rnn')

patn_evaluator.recognition(word_types, no_gating = True, n = 10, s_id = 's2')
recognition = patn_evaluator.conv_rec_dict()
durations = patn_evaluator.durations

cap_model = '../models/caption_model.16VQ'
img_model = '../models/image_model.16VQ'

patn_evaluator_VQ = evaluator(f_loc = data_loc, img_loc = img_model, 
                              cap_loc = cap_model, enc_type = 'rnn_VQ')

patn_evaluator_VQ.recognition(word_types, no_gating = True, n = 10, s_id = 's2')
recognition_VQ = patn_evaluator_VQ.conv_rec_dict()
durations_VQ = patn_evaluator_VQ.durations


# turn the recognition results into dataframes
p_10 = dict_to_df(recognition)
dur = dict_to_df(durations)
p_10vq = dict_to_df(recognition_VQ)

s2 = pd.concat([p_10, dur, p_10vq],1)
s2.columns = ['p@10', 'dur', 'p@10VQ']
s2['s_id'] = 's2'
s2['word_type'] = ['singular']*49 + ['plural']*42 + ['root']*49 +['third']*49 + ['participle']*49


p_10 = pd.DataFrame(p_10)
p_10.columns = ['p_10']
p_10['word_type'] = ['singular']*49 + ['plural']*42 + ['root']*49 +['third']*49 + ['participle']*49

p_10_VQ = pd.Series()
for x in recognition_VQ.keys():
    for y in recognition_VQ[x].keys():
        p_10_VQ = pd.concat([p_10_VQ, pd.Series(recognition_VQ[x][y], index = [y[:-2]])])

p_10_VQ = pd.DataFrame(p_10_VQ)
p_10_VQ.columns = ['p_10VQ']
p_10_VQ['word_type'] = ['singular']*49 + ['plural']*42 + ['root']*49 +['third']*49 + ['participle']*49

p_10 = pd.concat([p_10, p_10_VQ['p@10VQ']], 1)

s = pd.concat([s1, s2],0)

df = pd.DataFrame.join(df, s)

df['rel_dur'] = df['n_phones'] / df['dur']

df['log_freq'] = np.log(df['lemma_count'])
df.to_csv('./lmer_data.csv')


###############################################################################

df['nb_dens'] = pd.DataFrame.from_dict(n_density, orient = 'index', columns = ['nb_dens'])
ic = pd.DataFrame.from_dict(init_cohort, orient = 'index', 
                            columns = ['g1', 'g2','g3', 'g4', 'g5', 'g6',
                                       'g7', 'g8', 'g9', 'g10']
                            )

df = df.join(ic)
df = df.melt(df.columns[:16], var_name = 'gate', value_name = 'init_cohort')

def group_gating(gating_dict):
    grouped_gating = {}
    
    for key in gating_dict.keys():
        grouped = defaultdict(list)
        for w in gating_dict[key].keys():
            k = w.split('_')[0]
            grouped[k].append(gating_dict[key][w])
        grouped_gating[key] = grouped
    return grouped_gating
#################################### subject 1 ################################
patn_evaluator.recognition(word_types, s_id = 's1')  
gating_recognition = patn_evaluator.conv_rec_dict()

patn_evaluator_VQ.recognition(word_types, s_id = 's1')  
gating_recognition_VQ = patn_evaluator_VQ.conv_rec_dict()

gating_recognition_p10 = group_gating(gating_recognition)
gating_recognition_p10_VQ = group_gating(gating_recognition_VQ)

gating_df1 = pd.DataFrame()
for wt in gating_recognition_p10:
    g = pd.DataFrame.from_dict(gating_recognition_p10[wt], orient = 'index')
    g['word_type'] = wt
    gating_df1 = gating_df1.append(g)
gating_df1.columns = ['g1','g2','g3','g4','g5','g6','g7',
                    'g8','g9','word_type','g10']

gating_df_VQ1 = pd.DataFrame()
for wt in gating_recognition_p10_VQ:
    g = pd.DataFrame.from_dict(gating_recognition_p10_VQ[wt], orient = 'index')
    g['word_type'] = wt
    gating_df_VQ1 = gating_df_VQ1.append(g)
gating_df_VQ1.columns = ['g1','g2','g3','g4','g5','g6','g7',
                    'g8','g9','word_type','g10']


################################# subject 2 ###################################
patn_evaluator.recognition(word_types, s_id = 's2')  
gating_recognition = patn_evaluator.conv_rec_dict()

patn_evaluator_VQ.recognition(word_types, s_id = 's2')  
gating_recognition_VQ = patn_evaluator_VQ.conv_rec_dict()

gating_df2 = pd.DataFrame()
for wt in gating_recognition_p10:
    g = pd.DataFrame.from_dict(gating_recognition_p10[wt], orient = 'index')
    g['word_type'] = wt
    gating_df2 = gating_df2.append(g)
gating_df2.columns = ['g1','g2','g3','g4','g5','g6','g7',
                    'g8','g9','word_type','g10']

gating_df_VQ2 = pd.DataFrame()
for wt in gating_recognition_p10_VQ:
    g = pd.DataFrame.from_dict(gating_recognition_p10_VQ[wt], orient = 'index')
    g['word_type'] = wt
    gating_df_VQ2 = gating_df_VQ2.append(g)
gating_df_VQ2.columns = ['g1','g2','g3','g4','g5','g6','g7',
                    'g8','g9','word_type','g10']

gating_df1['s_id'] = 's1'
gating_df2['s_id'] = 's2'
gating = pd.concat([gating_df1, gating_df2], 0)
gating['word'] = gating.index
gating_VQ = pd.concat([gating_df_VQ1, gating_df_VQ2])
gating_VQ['word '] = gating_VQ.index
gating = gating.melt(['word', 's_id', 'word_type'], var_name = 'gate', 
            value_name = 'gatingp10')

gating= df.merge(x, how = 'inner')

gating['phones_seen'] = pd.Series([int(x[1:]) for x in  gating['gate']])
gating['rel_phones'] = gating['phones_seen']/ gating['n_phones']
gating.to_csv('./lmer_gating.csv')
