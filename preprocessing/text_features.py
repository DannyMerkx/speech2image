import string
import pickle
import sys
sys.path.append('/data/speech2image/preprocessing/coco_cleanup')

from text_cleanup import tokenise, correct_spel, remove_low_occurence, remove_stop_words, clean_wordnet, remove_punctuation, remove_numerical
from nltk.corpus import stopwords

def text_features_flickr(text_dict, output_file, append_name, node_list): 
    count = 1
    for node in node_list:
        print('processing file: ' + str(count))
        count+=1
        # create nodes for the raw text and the tokenised text
        raw_text_node = output_file.create_group(node, 'raw_text')
        token_node = output_file.create_group(node, 'tokens')
        # base name of the image caption pair to extract sentences from the dictionary
        base_name = node._v_name.split(append_name)[1]
        
        captions = text_dict[base_name]['sentences']
        for x in captions:
            
            raw = x['raw']
            tokens = x['tokens']
            output_file.create_array(raw_text_node, append_name + base_name + '_' + str(x['sentid']), bytes(raw, 'utf-8'))
            output_file.create_array(token_node, append_name +  base_name + '_' + str(x['sentid']), tokens) 

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def text_features_coco(text_dict, output_file, append_name, node_list): 
    # load the spelling correction dictionary
    stop_words = stopwords.words('english')
    spell_dict = load_obj('/data/speech2image/preprocessing/coco_cleanup/spell_dict')   
    coco_dict = load_obj('/data/speech2image/preprocessing/coco_cleanup/coco_dict')
    punct = string.punctuation
    digits = string.digits
    count = 1
    for node in node_list:
        print('processing file: ' + str(count))
        count+=1
        # create nodes for the raw text and the tokenised text
        raw_text_node = output_file.create_group(node, 'raw_text')
        token_node = output_file.create_group(node, 'cleaned_tokens')
        raw_token_node = output_file.create_group(node, 'raw_tokens')
        spell_token_node = output_file.create_group(node, 'spell_tokens')
        # base name of the image caption pair to extract sentences from the dictionary
        base_name = node._v_name.split(append_name)[1]
        
        captions = text_dict[base_name]        
        for x in captions:
            # the raw caption is just the original text, tokenised to remove extra spaces etc. and place a dot at the 
            # end of every sentence.            
            raw = ''.join([' ' + y for y in tokenise(x['caption'], lower = False)])[1:]
            if not raw[-1] == '.':
                raw = raw +' .'
            # raw tokens are the raw caption with only tokenisation
            raw_tokens = tokenise(x['caption'])
            if not raw_tokens[-1] == '.':
                raw_tokens[-1] = '.'
            # tokens with simple spelling correction applied. 
            spell_correct_tokens = correct_spel(raw_tokens, spell_dict)
            # cleaned tokens, with numericals removed, single occuring words removed, stop words removed and
            # only contains wordnet words. 
            cleaned_tokens =  clean_wordnet(remove_punctuation(remove_stop_words(remove_low_occurence(remove_numerical(raw_tokens, digits), coco_dict), stop_words),punct))
            
            output_file.create_array(raw_token_node, append_name +  base_name + '_' + str(x['id']), raw_tokens)
            output_file.create_array(spell_token_node, append_name +  base_name + '_' + str(x['id']), spell_correct_tokens)
            output_file.create_array(raw_text_node, append_name + base_name + '_' + str(x['id']), bytes(raw, 'utf-8'))
            output_file.create_array(token_node, append_name +  base_name + '_' + str(x['id']), cleaned_tokens) 
    
