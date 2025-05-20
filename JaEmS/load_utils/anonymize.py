import re
import copy
from .load_datasets import iterate_dataset


# https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex
def regex_anonymize(text, repls: dict[str]):
    reg = re.compile("(%s)" % "|".join(map(re.escape, repls.keys())))
    # For each match, look-up corresponding value in dictionary
    return reg.sub(lambda m: repls[m.group()], text) 


# can be rewritten to use format_conv_text
def anonymize_utt(utt, anonymizer):
    return {'speaker': utt['speaker'], 'text': anonymizer(utt['text'])}
        

def anonymize_conv(conv, speakers: tuple[str]=None, inplace=False):
    text = conv['text']
    if speakers is None:
        speakers = (text[0]['speaker'], text[1]['speaker'])
    repl = {speakers[0]: '', speakers[1]: ''}
    
    anonymizer = lambda x: regex_anonymize(x, repl)
    anon_conv = conv if inplace else copy.deepcopy(conv)
    anon_conv['text'] = [anonymize_utt(t, anonymizer) for t in text]
    return anon_conv


def anonymize_dataset(dataset, inplace=False):
    return iterate_dataset(dataset, anonymize_conv, inplace)
    