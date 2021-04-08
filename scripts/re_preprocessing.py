from typing import List
from anntools import Relation, Sentence, Collection, Keyphrase
from preprocessing import find_keyphrase_by_span

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spacy
import networkx as nx
# import es_core_news_sm

nlp = spacy.load('es_core_news_sm')


def find_relation(relations:List[Relation], entity1, entity2):
    for rel in relations:
        if rel.origin == entity1 and rel.destination == entity2 or \
           rel.origin == entity2 and rel.destination == entity1:
            return rel.label
    return 'empty'

def find_keyphrase_tokens(sentence:Sentence, doc:List):
    "Returns the spacy tokens of every keyphrase"
    text = sentence.text
    keyphrases = {}
    i = 0
    for token in doc:
        idx = text.index(token.text, i)
        n = len(token.text)
        i = idx + n
        keyphrase_id, _ = find_keyphrase_by_span(idx, idx+n, sentence.keyphrases, text)
        if keyphrase_id is None:
            continue
#         print(keyphrase_id, token)
        try:
            keyphrases[keyphrase_id].append(token) 
        except:
            keyphrases[keyphrase_id] = [token]
    return keyphrases    


def get_features(sentence:Sentence, doc:List):
    features = []
    keyphrases = find_keyphrase_tokens(sentence, doc)
    for i, keyphrase1 in enumerate(sentence.keyphrases):
        for keyphrase2 in sentence.keyphrases[i+1:]:
            try:
                tokens1 = keyphrases[keyphrase1.id]            
                tokens2 = keyphrases[keyphrase2.id]
            except:
                # This doesn't work properly because of the multitokens are not recognize
                pass
            features.append({
                'origin': keyphrase1.text,
                'destination': keyphrase2.text,
                'origin_tag': keyphrase1.label,
                'destination_tag': keyphrase2.label
                # Ideas:
                # el tamaño del camino del dependency graph entre los 2 tokens
                # quizá la secuencia entera a seguir codificada entre los 2 tokens principales
            })
    return features

def get_labels(sentence:Sentence, doc):
    labels = []
    for i, keyprhase1 in enumerate(sentence.keyphrases):
        for keyprhase2 in sentence.keyphrases[i+1:]:
#             print(keyprhase1.text, keyprhase2.text)
            labels.append(find_relation(sentence.relations, keyprhase1.id, keyprhase2.id)) 
    return labels


def get_instances(sentence):
    """
    Makes all the analysis of the sentence according to spacy
    """
    doc = nlp(sentence.text)
    features = get_features(sentence, doc)
    labels = get_labels(sentence, doc)
    return features, labels