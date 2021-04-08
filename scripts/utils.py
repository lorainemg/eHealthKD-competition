from typing import List
from anntools import Keyphrase


def find_keyphrase_by_span(i:int, j:int, keyphrases:List[Keyphrase], sentence:str, nlp):
    '''
    Returns the keyphrase id and the tag of a keyphrase based on the indices that a 
    token occupies in a given sentence
    '''
    # TODO: This has to handle correctly multitokens to work properly, so it should return a list of id / labels
    token = sentence[i:j]
    for keyphrase in keyphrases:
        for idx, (x, y) in enumerate(keyphrase.spans):
            word = sentence[x:y]
            if x <= i and y >= j:
                if idx == 0: return keyphrase.id, 'B-' + keyphrase.label
                else: return keyphrase.id, 'I-' + keyphrase.label 
            elif i <= x and j >= y:
                if idx == 0:
                    tokens = nlp.tokenizer(word)
                    if i + len(tokens[0]) <= j:
                        return keyphrase.id, 'B-' + keyphrase.label
                return keyphrase.id, 'I-' + keyphrase.label   
    return None, 'O'
