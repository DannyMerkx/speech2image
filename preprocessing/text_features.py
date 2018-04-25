import json
import tables

def text_features(text_path, output_file, append_name, node_list): 
    #open the .json file with all the text data
    with open(text_path) as data:
        data = json.load(data)
    data = data ['images']
    # convert the list of dictionaries from the json file to a dictionary of dictionaries
    text_dict = {}   
    for x in data:
        text_dict[x['filename'].strip('.jpg')] = x

    count = 1
    for node in node_list:
        print('processing file: ' + str(count))
        count+=1
        # create nodes for the raw text and the tokenised text
        try:    
            raw_text_node = output_file.create_group(node, 'raw_text')
            token_node = output_file.create_group(node, 'tokens')
        except:
            continue
        # base name of the image caption pair to extract sentences from the dictionary
        base_name = node._v_name.split(append_name)[1]
        
        captions = text_dict[base_name]['sentences']
        for x in captions:
            
            raw = x['raw']
            tokens = x['tokens']
            output_file.create_array(raw_text_node, append_name + base_name + '_' + str(x['sentid']), bytes(raw, 'utf-8'))
            output_file.create_array(token_node, append_name +  base_name + '_' + str(x['sentid']), tokens) 
    
