#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:13:04 2021
Test a trained model's word recognition performance.
@author: danny
"""
import torch
import tables
import pandas as pd
import sys
sys.path.append('/home/danny/Documents/project_code/speech2image/PyTorch/functions')
sys.path.append('/home/danny/Documents/project_code/speech2image/PyTorch/training_scripts')
from encoder_configs import create_encoders
from collections import defaultdict, Counter

class evaluator():
    def __init__(self, f_loc, img_loc, cap_loc, enc_type = None):
        # set cuda
        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor        
        
        self.features = tables.open_file(f_loc, mode='a')
        
        # load the data
        self.images = self.features.root.image._f_list_nodes()
        self.utterances = self.features.root.audio._f_list_nodes()
        # split by nouns and verbs
        self.nouns = []
        self.verbs = []
        for utt in self.utterances:
            if utt.word_type.read().decode() == 'noun':
                self.nouns.append(utt)
            elif utt.word_type.read().decode() == 'verb':
                self.verbs.append(utt)
        
        if enc_type != None:        
            self.load_encoder(enc_type, img_loc, cap_loc)
        
    def load_encoder(self, enc_type, img_loc, cap_loc):
        self.img_net, self.cap_net = create_encoders(enc_type)
        if self.dtype == torch.cuda.FloatTensor:
            cap_state = torch.load(cap_loc)
            img_state = torch.load(img_loc)            
        else:
            cap_state = torch.load(cap_loc, map_location = torch.device('cpu'))
            img_state = torch.load(img_loc, map_location = torch.device('cpu'))
    
        self.cap_net.load_state_dict(cap_state)
        self.img_net.load_state_dict(img_state)
        # set to eval model, critical to disable VQ network EMA updates
        self.cap_net.eval()
        self.img_net.eval()
        # disable gradients
        for param in self.cap_net.parameters():
            param.requires_grad = False
        for param in self.img_net.parameters():
            param.requires_grad = False
        
    # functions to embed the images and utterances
    def embed_images(self):
        image = self.dtype()
        # read and concatenate all image features
        for im in self.images:
            im = im.resnet.read()
            im = self.dtype(im)
            im.requires_grad = False
            image = torch.cat((image, im.unsqueeze(0).data))
        image = self.img_net(image)
        self.encoded_images = image.data

    def embed_utt(self, utt):
        l = [utt.shape[0]]
        utt = self.dtype(utt).t()
        utt.requires_grad = False
        return self.cap_net(utt.unsqueeze(0), l).data

    def cosine(self, emb_1, emb_2):
        return torch.matmul(emb_1, emb_2.t())

    # function for the p@n gating experiment, returns a dictionary with for 
    # each word how many correct images were retrieved
    def recognition(self, word_types, no_gating = False, n = 10, s_id = None):
        # dictionary which stores lemmas and their overall recognition scores
        self.rec_dict = {}
        self.durations = {}
        for t in word_types:
            self.rec_dict[t] = defaultdict(list)
            self.durations[t] = defaultdict(int)
        # embed the images
        self.embed_images()
        # embed the utterances, pick the top n most similar images and check
        # if they are correct
        for utt in self.utterances:
            word_form = utt.word_form.read().decode()
            # if a speaker id is given skip if this is not the right speaker
            if (s_id != None) and (utt.speaker_id.read().decode() != s_id):
                continue
            # similarly, if only certain word types are needed
            if word_form not in word_types:
                continue            

            # read the features and the phone alignment
            mfcc = utt.mfcc5.read()
            ali = utt.alignment_2.read()
            # if no gating is needed, use only the final gate i.e.
            # just using the whole utterance
            if no_gating:
                ali = [ali[-1]]
            for i, phon in enumerate(ali):
                # take at least 6 frames even if the first phone is shorter
                # because there is a 6 width conv kernel in the model.
                utt_emb = self.embed_utt(mfcc[:max(6,int(phon[1])),:])
                # generate the similarity matrix between this utterance and all
                # images, and look only at the n most similar images
                top_ims = self.cosine(utt_emb, self.encoded_images).topk(n)
                # get the lemma and the actual word form of the utterance
                
                transcript = utt.transcript.read().decode()
                self.rec_dict[word_form][f'{transcript}_{i}'].extend(self.correct_ims(top_ims, utt)) 
            self.durations[word_form][f'{transcript}_{i}'] = mfcc.shape[0]
        return self.rec_dict

    # function to load the top ims for an utterance and compare their annotations
    # to the utterance lemma and check if recognition is correct
    def correct_ims(self, top_ims, utt):
        correct = []
        # word type determines which img annotation to read, lemma determines
        # if recognition was correct
        word_type = utt.word_type.read().decode()   
        lemma = utt.lemma.read().decode()
        # read in the annotations and check if the target word is in 
        # the picture
        for idx in top_ims[1][0]:
            image = self.images[idx]
            if word_type == 'noun':
                annot = [x.decode() for x in image.nouns.read()]
                # for nouns, a counter to check whether the annotated object
                # occurs twice
                annot = Counter(annot)
            elif word_type == 'verb':
                annot = {x.decode():1 for x in image.verbs.read()}
            if lemma in annot.keys():
                correct.append(annot[lemma])
        return correct
    
    # convert the recognition lists into a single score per word
    def conv_rec_dict(self):
        self.recognition_scores = {word_type:{word: len(score) for word, 
                                              score in r_dict.items()
                                              } for word_type, 
                                   r_dict in self.rec_dict.items()
                                   }
        return self.recognition_scores
    #perform patn with random model
    def random_model(self, word_types, n = 5, model_type = 'rnn'):
        for x in range(n):
            img_net, cap_net = create_encoders('rnn')
            
            cap_net.eval()
            img_net.eval()
            # disable gradients
            for param in cap_net.parameters():
                param.requires_grad = False
            for param in img_net.parameters():
                param.requires_grad = False
            self.recognition(word_types, no_gating = True)
            self.conv_rec_dict()

# calculate overall p@n percentage from the recognition dict 
def p_at_n(rec_dict, n_subs):
    overall_p = 0
    overall_c = 0
    for d in rec_dict.keys():
        precision = 0
        n = 0   
        for x in rec_dict[d].keys():
            n+=1
            precision += rec_dict[d][x]
        print(f'precision@10 for {d}: {precision / (n * n_subs * 10)}')
        overall_p += precision
        overall_c += n
    print(f'overall precision@10: {overall_p/(overall_c * n_subs * 10)}')

# combine the patn call to generate the data for exp 1 in our paper
def patn_experiment(patn_eval, word_types): 
    # rec for s1
    patn_eval.recognition(word_types, no_gating = True, n = 10, s_id = 's1')
    print('p@10 for subject 1')
    p_at_n(patn_eval.conv_rec_dict(), 1)
    # rec for s2
    patn_eval.recognition(word_types, no_gating = True, n = 10, s_id = 's2')
    print('p@10 for subject 2')
    p_at_n(patn_eval.conv_rec_dict(), 1)
    # overall recognition
    patn_eval.recognition(word_types, no_gating = True, n = 10)
    print('overall P@10')
    p_at_n(patn_eval.conv_rec_dict(), 2)

def patn_overall(patn_eval, word_types): 
    # overall recognition
    patn_eval.recognition(word_types, no_gating = True, n = 10)
    print('overall P@10')
    p_at_n(patn_eval.conv_rec_dict(), 2)

def main():
    ######################## word recognition ################################    
    # location of the test utterance feature file
    data_loc = './test_features.h5'
    # word types can be tested seperately just in case
    word_types = ['singular', 'plural', 'root', 'third', 'participle']
    
    cap_model = '../models/caption_model.16'
    img_model = '../models/image_model.16'
    
    patn_evaluator = evaluator(f_loc = data_loc, img_loc = img_model, 
                               cap_loc = cap_model, enc_type = 'rnn')
    
    # this function combines all calls needed to evaluate P@10 per subject and
    # overall
    #patn_experiment(patn_evaluator, word_types)
    patn_overall(patn_evaluator, word_types)
     
    ############################ gated recogition #############################
    patn_evaluator.recognition(word_types)  
    gating_recognition = patn_evaluator.conv_rec_dict()
    ###################### word recognition VQ ###############################
    cap_model = '../models/caption_model.16VQ'
    img_model = '../models/image_model.16VQ'
    patn_evaluator_VQ = evaluator(f_loc = data_loc, img_loc = img_model, 
                                  cap_loc = cap_model, enc_type = 'rnn_VQ')
    
    # this function combines all calls needed to evaluate P@10 per subject and
    # overall
    patn_experiment(patn_evaluator_VQ, word_types)

    ############################ gated recogition #############################
    patn_evaluator_VQ.recognition(word_types)  
    gating_recognition_VQ = patn_evaluator_VQ.conv_rec_dict()
    ########################## naive baseline model #########################
    
    # load the verb and noun annotations
    noun_annot = pd.read_csv(open('../Image_annotations/annotations/Noun_annotations.csv'), 
                             index_col=0)
    verb_annot = pd.read_csv(open('../Image_annotations/annotations/Verb_annotations.csv'), 
                             index_col=0)
    # get the ten images with the highest number of total annotations and embed them
    top = (verb_annot.sum(1) + noun_annot.sum(1)).sort_values(ascending = False)[:10].index
    # replace the images in the evaluator with just the best 10
    patn_evaluator.images = [x for x in patn_evaluator.images if '_'.join(x._v_name.split('_')[1:])+'.jpg' in top]
    
    # now get the best possible recognition if the model always returned the same 10
    # images
    patn_evaluator.recognition(word_types, no_gating = True, n = 10, 
                               s_id = 's1')
    naive = patn_evaluator.conv_rec_dict()
    
    p_at_n(naive, 1)
    
    ############################## plurality experiment #######################
    
    # embed only nouns and keep track of plural and singular image annotations 
    # instead of total only
    recognition = gating(cap_net, nouns, encoded_images, images, 
                         ['singular', 'plural'], dtype, no_gating = True, n = 10, plural = True)
    # get only those nouns which had at least 10 images with plural annotation
    plural_nouns = (noun_annot[noun_annot == 2].sum(0) > 9) & (noun_annot[noun_annot == 1].sum(0) > 9)
    # drop shorst and sunglasses as those have no plural/singular target word
    plural_nouns = plural_nouns.drop('shorts')
    plural_nouns = plural_nouns.drop('sunglasses')
    plural_nouns = [x for x in plural_nouns.index if plural_nouns[x]]
    mapping = {'dogs': 'dog', 'men': 'man', 'boys': 'boy', 'girls': 'girl',
               'women': 'woman', 'shirts': 'shirt', 'balls': 'ball',
               'groups': 'group', 'rocks': 'rock', 'cameras': 'camera', 
               'bikes': 'bike', 'mountains': 'mountain', 'hat': 'hat', 
               'players': 'player', 'jackets': 'jacket', 'cars': 'car', 
               'buildings': 'building', 'dresses': 'dress', 'tables': 'table',
               'hands': 'hand', 'trees': 'tree', 'hills': 'hill', 
               'toys': 'toy', 'babies': 'baby', 'waves': 'wave',
               'benches': 'bench', 'sticks': 'stick', 'teams': 'team'}
    # get recognition scores for singular and plural targets
    plural = dict(recognition['plural'])
    singular = dict(recognition['singular'])
    
    pl_filtered = {}
    si_filtered = {}
    for key in plural:
        w = key.split('_')[0]
        if w in mapping.keys():
            w = mapping[w]
        C = Counter(plural[key])
        if w in plural_nouns:
            pl_filtered[key] = [C[1], C[2]]
            #pl_filtered[key] = C[2]/ max(C[1] + C[2], 0.0000001)
            print(f'{key}: {C[2]}')
    
    for key in singular:
        w = key.split('_')[0]
        C = Counter(singular[key])
        
        if  w in plural_nouns:
            si_filtered[key] = [C[1], C[2]]
            #si_filtered[key] = C[2]/ max(C[1] + C[2], 0.0000001)
            print(f'{key}: {C[2]}')
    
    pl_filtered = pd.DataFrame(pl_filtered).transpose()
    si_filtered = pd.DataFrame(si_filtered).transpose()        
    
    print(pl_filtered.sum(0)/pl_filtered.sum(0).sum())
    print(pl_filtered.sum(0))
    print(si_filtered.sum(0)/si_filtered.sum(0).sum())
    print(si_filtered.sum(0))
