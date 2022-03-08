#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:17:40 2021

@author: danny
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
import pingouin as pg
import logging

logging.basicConfig(filename='analysis.log', filemode='w', level=logging.DEBUG)

from collections import defaultdict

task_locs = {}
task_locs['simlex'] = './datasets/SimLex-999/SimLex-999.txt'
task_locs['MEN'] = './datasets//MEN/MEN_dataset_natural_form_full'
task_locs['wordsim_s'] = './datasets/wordsim/wordsim_similarity_goldstandard.txt'
task_locs['wordsim_r'] = './datasets/wordsim/wordsim_relatedness_goldstandard.txt'
task_locs['wordsim'] = './datasets/wordsim/wordsim353_agreed.txt'
task_locs['spp_lexdec'] = './datasets/SPP/data_lexdec.csv'
task_locs['spp_naming'] = './datasets/SPP/data_naming.csv'
task_locs['subtlex'] = './datasets/subtlex.txt'
task_locs['RareWords'] = './datasets/rw/rw.txt'

vec_locs = []
vec_locs.append(['VGE', './word_vectors/grounded_vecs.txt'])
vec_locs.append(['Word2Vec', './word_vectors/word2vec.txt'])
vec_locs.append(['FastText', './word_vectors/fasttext.txt'])
vec_locs.append(['Glove', './word_vectors/glove.txt'])
vec_locs.append(['Word2Vec_pretrained', './word_vectors/word2vec_pretrained.txt'])
vec_locs.append(['FastText_pretrained', './word_vectors/fasttext_pretrained.txt'])
vec_locs.append(['Glove_pretrained', './word_vectors/glove_pretrained.txt'])
class vector_sim_tester():
    def __init__(self, vec_locs, task_locs):
        # load the word vectors you want to test
        self.vecs = {}
        for vec in vec_locs:
            self.vecs[vec[0]] = self.read_vecs(vec[1])
            
        self.task_locs = task_locs
        self.results = defaultdict(dict)
        
    def analysis(self):  
        # run the word similarity evaluations
        self.wordsim()
        self.simlex()    
        self.men()
        self.rarewords()
        self.spp()
    # function to read text files with word vectors (standard word2vec format)
    def read_vecs(self, file_loc):      
        file = open(file_loc)
        vecs = [x.split() for x in file.read().split('\n') if not x == '']  
        vecs = {v[0]: np.array([float(x) for x in v[1:]]) for v in vecs}
        # make sure all vectors are normalised, this improves performance
        for v in vecs.keys():
            vecs[v] = vecs[v] / np.sqrt(sum(vecs[v]**2))
        file.close()
        return vecs
    # function to get all single character edits for a word. Used to prepare
    # control variables for SPP
    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
    
    # print/save the correlation of vector similarities to human similarity 
    # annotations
    def corr(self, data, col1, test_name):  
        for v in self.vecs.keys():
            n = data.dropna().shape[0]
            pearson = stats.pearsonr(data.dropna()[col1], data.dropna()[v])
            logging.info(f'{v} {test_name} pearson r: R = {pearson[0]}, P = {pearson[1]}, n = {n}')
            self.results[test_name][v] = (pearson[0], pearson[1])
            spearman = stats.spearmanr(data.dropna()[col1], data.dropna()[v])
            logging.info(f'{v} {test_name} Spearman r: R = {spearman[0]}, P = {spearman[1]}, n = {n}')
        
        #print(pg.partial_corr(data = data, x = col1, y = 'grounded', covar = col2))
    
    def partial_corr(self, data, x, y, test_name):
        for v in self.vecs.keys():
            if not v == y:
                z =  [v, v.split('_')[0]]
                p = pg.partial_corr(data.dropna(), x, y, covar = z)
                logging.info(f'{test_name} partial correlation: y = {y}, cov = {v}, R = {p["r"][0]}, P = {p["p-val"][0]}, n = {p["n"]}')
                self.results[test_name][f'{v}_partial'] = (p["r"][0], p["p-val"][0])
                
    # Takes a pandas file of word pairs with a human similarity rating, and 
    # calculates the vector similarities for the pairs for all vector types.
    def get_sims(self, data, wordcol1, wordcol2):
        for idx in range(0, len(data)):
            row  = data.iloc[idx]
            word1 = row[wordcol1]
            word2 = row[wordcol2] 
            for v in self.vecs.keys():
                try:
                    data.loc[idx, v] = np.matmul(self.vecs[v][word2], 
                                                 self.vecs[v][word1])
                except:
                    continue
        return data
    
    # prepare the similarity evaluation sets, by loading them into pandas 
    # dataframes, adding the right column names and the vector similarities
    def prep_sim_data(self, task_name, header = 'infer', sep = '\t', 
                       columns = None, skiprows = None):
        data = pd.read_table(self.task_locs[task_name], header = header, 
                             sep = sep, skiprows = skiprows)
        if columns:
            data.columns = columns
        for v in self.vecs.keys():
            data[v] = np.nan
        data = self.get_sims(data, 'word1', 'word2')
        return data
    
    def prep_spp(self, name):
        # read the subtlex data
        subtlex= pd.read_csv(self.task_locs['subtlex'], sep = '\t')
        subtlex_dict = defaultdict(int)
        subtlex_cd = defaultdict(int)
        # get frequencies and contextual counts
        for idx in range(0, len(subtlex)):
            row = subtlex.iloc[idx]
            word = str(row['Word']).lower()
            subtlex_dict[word] += row['FREQcount']         
            subtlex_cd[word] += row['CDcount']
        # read spp data    
        spp = pd.read_csv(self.task_locs[name])
        # add subtlex data to SPP 
        for idx in range(0, len(spp)):
            row  = spp.iloc[idx]
            prime = row['prime']
            target = row['target']
    
            spp.loc[idx, 'prime_len'] = len(prime)
            spp.loc[idx, 'target_len'] = len(target)
            spp.loc[idx, 'prime_freq'] = np.log(subtlex_dict[prime]+1) 
            spp.loc[idx, 'target_freq'] = np.log(subtlex_dict[target]+1)
            spp.loc[idx, 'prime_cd'] = np.log(subtlex_cd[prime]+1) 
            spp.loc[idx, 'target_cd'] = np.log(subtlex_cd[target]+1)
            spp.loc[idx, 'prime_nb'] = len([x for x in self.edits1(prime) if x in subtlex_dict.keys()])
            spp.loc[idx, 'target_nb'] = len([x for x in self.edits1(target) if x in subtlex_dict.keys()])
        # add the vector similarities
        for v in self.vecs.keys():
            spp[v] = np.nan
        spp = self.get_sims(spp, 'prime', 'target')      
        
        spp['task'] = name 
        spp['isi'] = spp['isi'].factorize()[0]
        # normalise RT
        temp = spp[spp['isi'] == 0]['meanRT']
        short = ( temp - temp.mean())/temp.std()
        temp = spp[spp['isi'] == 1]['meanRT']
        long = ( temp - temp.mean())/temp.std()
        spp['normRT'] = pd.concat([short, long])
        return spp    
    ####################### word similarity evaluations #######################
    def simlex(self):
        simlex = self.prep_sim_data('simlex')
        self.corr(simlex, 'SimLex999', 'SimLex999')    
        self.partial_corr(simlex, 'SimLex999', 'VGE', 'SimLex999')
        
        concr = simlex[simlex['concQ'] == 4]
        self.corr(concr, 'SimLex999', 'SimLex999 Q4')
        self.partial_corr(concr, 'SimLex999', 'VGE', 'SimLex999 Q4')        
        
        concr = simlex[simlex['concQ'] == 1]
        self.corr(concr, 'SimLex999',  'SimLex999 Q1')
        self.partial_corr(concr, 'SimLex999', 'VGE', 'SimLex999 Q1')  
        
        #simassoc = simlex[simlex['SimAssoc333'] == 1]
        #self.corr(simassoc, 'SimLex999', 'SimAssoc333')
        #self.partial_corr(simassoc, 'SimLex999', 'VGE', 'SimAssoc333')  
        
    def wordsim(self):
        ws = self.prep_sim_data('wordsim_s', header = None, 
                                 columns = ['word1', 'word2', 'sim'])
        self.corr(ws, 'sim', 'WordSim-S')
        self.partial_corr(ws, 'sim', 'VGE', 'WordSim-S')  
        
        ws = self.prep_sim_data('wordsim_r', header = None, 
                                 columns = ['word1', 'word2', 'sim'])    
        self.corr(ws, 'sim', 'WordSim-R')
        self.partial_corr(ws, 'sim', 'VGE', 'WordSim-R')
        
        ws = self.prep_sim_data('wordsim', header = None, skiprows = 11,
                                 columns = ['relationship', 'word1', 'word2', 
                                            'sim'
                                            ],
                                 )    
        self.corr(ws, 'sim', 'WordSim353')
        self.partial_corr(ws, 'sim', 'VGE', 'WordSim353')
        
    def men(self):
        men = self.prep_sim_data('MEN', header = None, sep = ' ', 
                                  columns = ['word1', 'word2', 'sim']) 
        self.corr(men, 'sim', 'MEN')
        self.partial_corr(men, 'sim', 'VGE', 'MEN')
        
    def rarewords(self):
        rw = self.prep_sim_data('RareWords', header= None, 
                                 columns = ['word1', 'word2', 'sim', 'annot1', 
                                            'annot2', 'annot3', 'annot4', 
                                            'annot5', 'annot6', 'annot7', 
                                            'annot8', 'annot9', 'annot10'
                                            ]
                                 )
        self.corr(rw, 'sim', 'RareWords')
        self.partial_corr(rw, 'sim', 'VGE', 'RareWords')
        
    ###################### SPP evaluations ####################################    
    def spp_baseline(self, spp):
        spp = spp[(spp['target_freq'] > 0)]   
        regr =  smf.ols('normRT ~  target_len + target_freq + target_nb \
                        + target_cd', 
                        data=spp.dropna()).fit()
        logging.info('Baseline model')
        logging.info(regr.aic)
        logging.info(regr.llf)
        logging.info(regr.summary())
        
    def spp_sims(self, spp):
        spp = spp[(spp['target_freq'] > 0)]
        for v in self.vecs.keys():
            regr =  smf.ols(f'normRT ~ target_len +  target_freq + target_nb +\
                            target_cd + {v} * isi + {v} * task', 
                            data=spp.dropna()).fit()  
            logging.info(f'{v}')
            logging.info(regr.aic)
            logging.info(regr.llf)
            logging.info(regr.summary())
            
    def spp_sims_nested(self, spp, vec_name):        
        spp = spp[(spp['target_freq'] > 0)]
        for v in self.vecs.keys():
            if not v == vec_name:
                if 'pretrained' in v:
                    x = v.split('_')[0]
                    regr =  smf.ols(f'normRT ~ target_len + target_freq + \
                                    target_nb + target_cd + {v} * isi * task\
                                    + {x} * isi * task', 
                                    data=spp.dropna()).fit()
                    logging.info(f'{v} + {x}')
                    logging.info(regr.llf)
                    logging.info(regr.summary())
                    
                    d = spp.dropna()
                    d['resid'] = regr.resid
                    
                    regr =  smf.ols(f'resid ~ {vec_name} * isi * task', 
                                    data=d.dropna()).fit()                
                    logging.info(f'{v} + {x} + {vec_name} residual model ')
                    logging.info(regr.llf)
                    logging.info(regr.summary())
                    
                regr =  smf.ols(f'normRT ~ target_len + target_freq + \
                                target_nb + target_cd + {v} * isi + {v} * task', 
                                data=spp.dropna()).fit()
                logging.info(f'{v} + {vec_name}')
                logging.info(regr.llf)   
                d = spp.dropna()
                d['resid'] = regr.resid
                
                regr =  smf.ols(f'resid ~ {vec_name} * isi * task', 
                                data=d.dropna()).fit()                
                logging.info(f'{v} + {vec_name} residual model ')
                logging.info(regr.llf)
                logging.info(regr.summary())                

    def spp_sims_nested2(self, spp, vec_name):        
        spp = spp[(spp['target_freq'] > 0)]
        for v in self.vecs.keys():
            if not v == vec_name:
                if 'pretrained' in v:
                    x = v.split('_')[0]
                    regr =  smf.ols(f'normRT ~ target_len + target_freq + \
                                    target_nb + target_cd + {v} * isi + {v}* task\
                                    + {x} * isi + {x} * task', 
                                    data=spp.dropna()).fit()
                    logging.info(f'{v} + {x}')
                    logging.info(regr.llf)
                    logging.info(regr.summary())

                    
                    regr2 =  smf.ols(f'normRT ~ target_len + target_freq + \
                                    target_nb + target_cd + {v} * isi + {v}* task\
                                    + {x} * isi + {x} * task + {vec_name} * isi + {vec_name}* task', 
                                    data=spp.dropna()).fit() 
                        
                    logging.info(f'{v} + {x} + {vec_name}')
                    logging.info(regr2.llf)
                    logging.info(regr2.summary())
                    logging.info(regr.llf - regr2.llf)
                else:    
                    regr =  smf.ols(f'normRT ~ target_len + target_freq + \
                                    target_nb + target_cd + {v} * isi + {v}* task', 
                                    data=spp.dropna()).fit()
                    logging.info(f'{v} + {vec_name}')
                    logging.info(regr.summary())
                    logging.info(regr.llf)   
 
                    regr2 =  smf.ols(f'normRT ~ target_len + target_freq + \
                                    target_nb + target_cd + {v} * isi + {v}* task +\
                                    {vec_name} * task + {vec_name}* isi', 
                                    data=spp.dropna()).fit() 
                    
                    logging.info(f'{v} + {vec_name} residual model ')
                    logging.info(regr2.llf)
                    logging.info(regr2.summary())
                    logging.info(regr.llf - regr2.llf)
        
    def spp(self):
        # prepare the spp dataset
        spp_lexdec = self.prep_spp('spp_lexdec') 
        spp_naming = self.prep_spp('spp_naming') 
        spp = pd.concat([spp_lexdec, spp_naming])    
        self.spp_baseline(spp)
        
        self.spp_sims(spp)

        self.spp_sims_nested2(spp, 'VGE')                      
        
tester = vector_sim_tester(vec_locs, task_locs)
tester.analysis()         
   