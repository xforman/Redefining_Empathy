import json
import os
import glob
import copy




def load_json(path: str):
    """
    Load JSON data from a file.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    data : dict
        Parsed JSON data.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data

    
#----------------- Hugging face Chat_template Utils ---------------------#
def utt2hf(name, utt, i):
    """
    Convert an utterance to Hugging Face chat format.

    Parameters
    ----------
    name : str
        Speaker name.
    utt : str
        Utterance text.
    i : int
        Index of the utterance.

    Returns
    -------
    formatted_utt : dict
        Formatted utterance with 'role' and 'content'.
    """
    return {'role': 'user' if i % 2 == 0 else 'assistant',
            'content': utt}

    
# expects the utterance in the format {'speaker': name, 'text': text}
# format_utt is a function taking two str args
# is inplace   
def format_conv_text(conv, format_utt, inplace=True):
    """
    Format a conversation in place using a provided formatting function.

    Parameters
    ----------
    conv : list of dict
        List of utterances in the format {'speaker': str, 'text': str}.
    format_utt : function
        Function to format each utterance.
    inplace : bool, optional
        Whether to modify the conversation in place. Default is True.

    Returns
    -------
    conv : list of dict
        The formatted conversation.
    """
    for i, utt in enumerate(conv):
        conv[i] = format_utt(utt['speaker'], utt['text'], i)
    return conv


def text2hf_template(conv, inplace=True):
    """
    Convert a conversation dict to Hugging Face format.

    Parameters
    ----------
    conv : dict
        Dictionary with a 'text' key containing a conversation list.
    inplace : bool, optional
        Whether to modify the input in place. Default is True.

    Returns
    -------
    conv : dict
        Modified conversation with utterances in HF format.
    """
    conv = conv if inplace else copy.deepcopy(conv)
    format_conv_text(conv['text'], utt2hf, inplace)
    if conv['text'][-1]['role'] != 'assistant':
        conv['text'].pop()
    return conv


def format_dataset_conv(dataset, utt_format, inplace=False):
    """
    Format all conversations in a dataset using a given function.

    Parameters
    ----------
    dataset : dict
        Dataset of conversations.
    utt_format : function
        Formatting function to apply to each conversation.
    inplace : bool, optional
        Whether to modify the dataset in place. Default is False.

    Returns
    -------
    formatted_dataset : dict
        Formatted dataset.
    """
    return iterate_dataset(dataset, utt_format, inplace)
    

def conv_info_templ(text='', creation_info='', utils=''):
    """
    Create a conversation metadata dictionary.

    Parameters
    ----------
    text : str or list, optional
        The conversation text. Default is empty string.
    creation_info : str, optional
        Information about the conversation's origin. Default is empty string.
    utils : dict or str, optional
        Additional metadata utilities. Default is empty string.

    Returns
    -------
    info : dict
        Conversation metadata dictionary.
    """
    return {'text': text, 'creation_info': creation_info, 'utils': utils}


#------------------ Bias Dataset Loading ---------------------#
def bias_dataset_utt_template(utt, speaker_name):
    """
    Create a bias dataset utterance entry.

    Parameters
    ----------
    utt : str
        Utterance text.
    speaker_name : str
        Name of the speaker.

    Returns
    -------
    entry : dict
        Dictionary with 'speaker' and 'text' keys.
    """
    return {'speaker': speaker_name, 'text': utt}


def text2convBD(text, utt_template):
    """
    Convert raw text into a conversation format for bias datasets.

    Parameters
    ----------
    text : str
        Raw conversation text.
    utt_template : function
        Function to convert utterance to dict.

    Returns
    -------
    conv : list of dict
        List of formatted utterances.
    """
    split_text = text.replace('\n\n', '\n').split('\n')
    conv = [utt_template(utt, f'speaker_{i%2+1}') for i, utt in enumerate(split_text)] 
    return conv
    

def iterate_dataset(dataset, func, inplace=False):
    """
    Apply a function to all conversations in a dataset.

    Parameters
    ----------
    dataset : dict
        Dataset of conversations.
    func : function
        Function to apply to each conversation.
    inplace : bool, optional
        Whether to apply changes in place. Default is False.

    Returns
    -------
    iter_dataset : dict
        Modified dataset.
    """
    iter_dataset = dataset if inplace else copy.deepcopy(dataset)
    
    for soc_grp, prot_attrs in iter_dataset.items():
        for prot_attr, attr_info in prot_attrs.items():
            for conv_info, conv in attr_info['conversations'].items():
                func(conv, inplace=True)
                    
    return iter_dataset


def load_bias(bias_attr_path, text2conv=text2convBD, json_path=None, creation_info=''):
    """
    Load a bias dataset from a grounding file.

    Parameters
    ----------
    bias_attr_path : str
        Path to the grounding JSON file.
    text2conv : function, optional
        Function to convert raw text to conversation. Default is `text2convBD`.
    json_path : str or None, optional
        If specified, save the dataset to this path. Default is None.
    creation_info : str, optional
        Metadata to store as creation info in each conversation. Default is empty.

    Returns
    -------
    bias_dataset : dict
        Constructed bias dataset.
    """
    bias_attrs_grounding = load_json(bias_attr_path)
    
    bias_dataset = {}
    for conv_data in bias_attrs_grounding:
        soc_grp, attr, grounding = conv_data['metadata']
        if bias_dataset.get(soc_grp, None) is None:
            bias_dataset[soc_grp] = {}
        if bias_dataset[soc_grp].get(attr, None) is None:
            bias_dataset[soc_grp][attr] = {'value': attr, 'conversations': {}}
            
        attr_data = bias_dataset[soc_grp][attr]
        conv = text2conv(conv_data['conversation'], bias_dataset_utt_template)
        utils = {'prompt': conv_data['input'], 'source': conv_data['source'], 
                 'grounding': grounding}
        conv_info =  conv_info_templ(conv, creation_info, utils)
        attr_data['conversations'][grounding] = conv_info
    
    if json_path is not None:
        with open(json_path, 'w') as f:
            json.dump(bias_dataset, f)
        
    return bias_dataset


#------------------ Locomo Dataset Loading -------------------#
def locomo2bias_dataset(locomo_data, conv_templ):
    """
    Convert Locomo dataset to the bias dataset format.

    Parameters
    ----------
    locomo_data : list
        Raw Locomo dataset loaded from JSON.
    conv_templ : function
        Function to apply to each session conversation.

    Returns
    -------
    bias_dataset : dict
        Reformatted dataset under 'test_samples' key.
    """
    dataset_dict = {i: {'value': i, 'conversations': {}}
                    for i in range(len(locomo_data))}
    for i, sessions_info in enumerate(locomo_data):
        session_convs = sessions_info['conversation']
        sessions = len(sessions_info['event_summary'])
        
        for x in range(1, sessions + 1):
            sess_name = f'session_{x}'
            # restructure/butcher the locomo's format into the one we use
            restr_conv = [{'speaker': utt['speaker'], 'text': utt['text']}
                          for utt in session_convs[sess_name]]
            dataset_dict[i]['conversations'][sess_name] = conv_templ(restr_conv)
    
    return {'test_samples': dataset_dict}


def load_locomo(locomo_path, conv_templ, json_path=None):
    """
    Load and convert Locomo dataset from JSON format.

    Parameters
    ----------
    locomo_path : str
        Path to the Locomo dataset file.
    conv_templ : function
        Function to apply to each conversation.
    json_path : str or None, optional
        If specified, save the dataset to this path. Default is None.

    Returns
    -------
    dataset : dict
        Reformatted dataset in bias format.
    """
    with open(locomo_path, 'r') as f:
        locomo_data = json.load(f)  
    
    dataset = locomo2bias_dataset(locomo_data, conv_templ)

    if json_path is not None:
        with open(json_path, 'w') as f:
            json.dump(dataset, f)

    return dataset


def load_n_tuples(sample_dataset, n_tuple_path):
    """
    Load and format n-tuples from a file.

    Parameters
    ----------
    sample_dataset : dict
        Dataset from which conversations will be sampled.
    n_tuple_path : str
        Path to the JSON file containing n-tuples.

    Returns
    -------
    n_tuple_dataset : dict
        Dataset formatted with n-tuples.
    n_tuples_dict : dict
        Dictionary of n-tuple groupings.
    """
    n_tuples = load_json(n_tuple_path)
        
    n_tuples_dict = {}
    test_sample = {}
    c = 'conversations'
    for ng_id, nt_group in enumerate(n_tuples):
        conv_id = nt_group['conv_id'] - 1
        speaker = nt_group['speaker']
        test_sample[ng_id] = {'value': f'{speaker}_{conv_id}', c: {}}
        n_tuples_dict[ng_id] = nt_group['n-tuple']
        
        for i, sess_i in enumerate(nt_group['diag_order']):
            conv = sample_dataset[conv_id][c][f'session_{sess_i}']['text']
            conv = conv if conv[0]['speaker'] == speaker else conv[1:]
            session_info = {'text': conv, 'creation_info': 
                       f'locomo conv {conv_id}, session {sess_i}',
                       'utils': {'speaker': speaker}}
            test_sample[ng_id][c][f'session_{i}'] = session_info

    return {'test_samples': test_sample}, n_tuples_dict
    