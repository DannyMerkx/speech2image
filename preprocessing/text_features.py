import tables
import string

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
    
def text_features_coco(text_dict, output_file, append_name, node_list): 
    count = 1
    for node in node_list:
        print('processing file: ' + str(count))
        count+=1
        # create nodes for the raw text and the tokenised text
        raw_text_node = output_file.create_group(node, 'raw_text')
        token_node = output_file.create_group(node, 'tokens')
        # base name of the image caption pair to extract sentences from the dictionary
        base_name = node._v_name.split(append_name)[1]
        
        captions = text_dict[base_name]
        for x in captions:
            
            raw = x['caption']
            # separate the dot at the end of sentence from the last word
            if raw[-1] == '.':
               raw = raw[:-1] + ' .'
            # tokenize the input (remove punctuation, make lower case and split into words)
            tokens = raw.lower()
            for y in string.punctuation:
                tokens = tokens.replace(y, '')
            tokens = tokens.split(' ')
            output_file.create_array(raw_text_node, append_name + base_name + '_' + str(x['id']), bytes(raw, 'utf-8'))
            output_file.create_array(token_node, append_name +  base_name + '_' + str(x['id']), tokens) 
    
