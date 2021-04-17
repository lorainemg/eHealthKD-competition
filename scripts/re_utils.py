from typing import List
from anntools import Relation, Sentence, Collection
from utils import find_keyphrase_by_span

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spacy

# import es_core_news_sm

nlp = spacy.load('es_core_news_sm')


def get_keyphrases_pairs(keyphrases):
    """Makes keyphrases pairs to extract the relation"""
    return [(keyphrase1, keyphrase2) for i, keyphrase1 in enumerate(keyphrases) for keyphrase2 in keyphrases[i + 1:]]


################################ Preprocessing ################################
# Preprocess the features, getting the training instances (features + labels)

def find_relation(relations: List[Relation], entity1: int, entity2: int):
    """
    Given the ids of two keyphrases, returns the label of the entity conecting them
    """
    # Ahora que lo pienso no estoy segura si entre 2 entidades puede establecerse más de una relación
    # En la práctica, anotando, nunca me pasó
    for rel in relations:
        if rel.origin == entity1 and rel.destination == entity2 or \
                rel.origin == entity2 and rel.destination == entity1:
            return rel.label
    return 'empty'


def find_keyphrase_tokens(sentence: Sentence, doc: List):
    """Returns the spacy tokens corresponding to every keyphrase"""
    text = sentence.text
    keyphrases = {}
    i = 0
    for token in doc:
        idx = text.index(token.text, i)
        n = len(token.text)
        i = idx + n
        keyphrase_id, _ = find_keyphrase_by_span(idx, idx + n, sentence.keyphrases, text, nlp)
        if keyphrase_id is None:
            continue
        #         print(keyphrase_id, token)
        try:
            keyphrases[keyphrase_id].append(token)
        except:
            keyphrases[keyphrase_id] = [token]
    return keyphrases


def get_features(sentence: Sentence, doc: List):
    """
    For every pair of keyphrases, its features are returned.
    """
    features = []
    keyphrases = find_keyphrase_tokens(sentence, doc)
    for keyphrase1, keyphrase2 in get_keyphrases_pairs(sentence.keyphrases):
        try:
            tokens1 = keyphrases[keyphrase1.id]
            tokens2 = keyphrases[keyphrase2.id]
        except:
            # This doesn't work properly because the multitokens are not recognize
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


def get_labels(sentence: Sentence, doc):
    """
    Returns the label if the relation between every pair of keyphrases.
    For the pairs of keyphrases with no relation, 'empty' is returned.
    """
    labels = []
    for keyphrase1, keyphrase2 in get_keyphrases_pairs(sentence.keyphrases):
        labels.append(find_relation(sentence.relations, keyphrase1.id, keyphrase2.id))
    return labels


def get_instances(sentence: Sentence, labels=True):
    """
    Makes all the analysis of the sentence according to spacy preprocessing.
    Returns the features and the labels corresponding to those features in the sentence.
    """
    doc = nlp(sentence.text)
    features = get_features(sentence, doc)
    if labels:
        labels = get_labels(sentence, doc)
        return features, labels
    else:
        return features


################################ Postprocessing ################################
# Postprocess the labels, converting the output of the classifier in the expected manner

def postprocessing_labels(labels, indices, collection: Collection):
    for sent_label, index in zip(labels, indices):
        sentence = collection[index]
        keyphrases = get_keyphrases_pairs(sentence.keyphrases)
        for label, (origin, destination) in zip(sent_label, keyphrases):
            if label != 'empty':
                relation = Relation(sentence, origin.id, destination.id, label)
                sentence.relations.append(relation)
