#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:34:53 2020

@author: danny
"""
import glob
import nltk
from collections import defaultdict

kaldi_loc = '/home/danny/kaldi/egs/s2i/'
audio_loc = 's2i_audio/train/danny/'
audio_files = glob.glob(f'{kaldi_loc}{audio_loc}*.wav')

script = '/home/danny/Documents/databases/Flickr_Test/word_recordings/script.txt'
file = open(script)

words = []
for line in file:
    words.extend(line.split())
    
dictionary = nltk.corpus.cmudict.dict()

dictionary['skateboards'] = [['S', 'K', 'EY1', 'T', 'B', 'AO2', 'R', 'D', 'Z']]
dictionary['surfs'] = [['S', 'ER1', 'F', 'S']]
transcripts = []
for word in words:
    transcripts.append(dictionary[word])
    
    
wav_scp = {}

for file in audio_files:
    idx = file.split('/')[-1].split('.')[0]
    wav_scp[f'{idx}_{words[int(idx)-1]}'] = [file, dictionary[words[int(idx)-1]], words[int(idx)-1]]
    
    
file_loc = f'{kaldi_loc}/data/train/'
# create the required kaldi files

# spk2gender
file = open(file_loc + 'spk2gender', 'w')
file.write('danny m')
file.close()

# wav.scp
file = open(file_loc + 'wav.scp', 'w')
for key in wav_scp.keys():
    file.write(f'{key} {wav_scp[key][0]}\n')
file.close()

# text
file = open(file_loc + 'text', 'w')
for key in wav_scp.keys():
    file.write(f'{key} {wav_scp[key][2].upper()}\n')
file.close()

# utt2spk
file = open(file_loc + 'utt2spk', 'w')
for key in wav_scp.keys():
    file.write(f'{key} danny\n')
file.close()

file_loc = f'{kaldi_loc}/data/local/lang/'
# language files
words = []
file = open(file_loc + 'lexicon.txt', 'w')
file.write('!SIL SIL\n')
file.write('<oov> <oov>\n')
for key in wav_scp.keys():
    for pron in wav_scp[key][1]:
        pron = ' '.join(pron)
        if not f'{wav_scp[key][2].upper()} {pron}' in words:
            file.write(f'{wav_scp[key][2].upper()} {pron}\n')
            words.append(f'{wav_scp[key][2].upper()} {pron}')
file.close()

#non-silence phones
file = open(file_loc + 'nonsilence_phones.txt', 'w')
phones = []
stressed = defaultdict(list)
for key in wav_scp.keys():
    for pron in wav_scp[key][1]:
        for phon in pron:
            if (('0' in phon) or ('1' in phon) or ('2' in phon)):
                if not phon in stressed[phon[:-1]]:
                    stressed[phon[:-1]].append(phon)            
            elif not phon in phones:
                file.write(f'{phon}\n')
            phones.append(phon)

for key in stressed.keys():
    for phon in stressed[key]:
        file.write(f'{phon} ')
    file.write('\n')
file.close()

# silence phones
file = open(file_loc + 'silence_phones.txt', 'w')
file.write('sil\n')
file.write('<oov>\n')
file.close()

# optional silence
file = open(file_loc + 'optional_silence.txt', 'w')
file.write('sil\n')
file.close()

