import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spacy
import networkx as nx
from anntools import Sentence, Keyphrase
from utils import find_keyphrase_by_span


nlp = spacy.load('es_core_news_sm')

sufixes = nlp.Defaults.suffixes + [r'%',r'\.']
suffix_re = spacy.util.compile_suffix_regex(sufixes)
nlp.tokenizer.suffix_search = suffix_re.search

################################ Preprocessing ################################
# Preprocess the features, getting the training instances (features + labels)

def get_features(tokens):
    '''
    Given a list of tokens returns the features of those tokens.
    They will be used for trainig in the first task (Name Entity Recognition)'''
    graph = nx.DiGraph()
    digraph = nx.DiGraph()
    features = []
    for token in tokens:
        # Constructs the dependency graph, both not directed and directed
        # Right now its not being used for nothing
        # Creates and edge for each token's children
        for child in token.children:
            digraph.add_edge(token.i, child.i)
            graph.add_edge(token.i, child.i, attr_dict={'dir': '/'})
            graph.add_edge(child.i, token.i, attr_dict={'dir': '\\'})
            # edges.append((token.i, child.i))
        features.append({
            'dep': token.dep_,
            'pos': token.pos_,
            'lemma': token.lemma_
        })
    return features


def get_labels(tokens, sentence:Sentence):
    '''
    Given a list of tokens and the sentences containing them returns the labels of those tokens.
    They will be used for trainig in the first task (Name Entity Recognition)
    '''
    text = sentence.text
    instances = {}
    i = 0
    for token in tokens:
        idx = text.index(token.text, i)
        n = len(token.text)
        i = idx + n 
        # instances[(idx, idx + n)] = classify_as_keyphrase(token.text, sentence.keyphrases, idx)
        _, instances[(idx, idx + n)] = find_keyphrase_by_span(idx, idx+n, sentence.keyphrases, sentence.text, nlp)
    return instances.values()


def get_instances(sentence:Sentence, labels=True):
    """
    Makes all the analysis of the sentence according to spacy.
    Returns the features and the labels correspinding to those features in the sentence.    
    """
    # The tokens and its features are extracted with spacy
    doc = nlp(sentence.text)
    features = get_features(doc)
    if labels:
        labels = get_labels(doc, sentence)
        return features, list(labels)
    else:
        return features

################################ Postprocessing ################################
# Postprocess the labels, converting the output of the classifier in the expected manner

def postprocessing_labels(labels, indices, sentences):
    next_id = 0
    for sent_label, index in zip(labels, indices):
        multiple_concepts = []
        multiple_actions = []
        multiple_predicates = []
        multiple_references = []
        sent = sentences[index]
        tokens = nlp.tokenizer(sent.text)
        for label, word in zip(sent_label, tokens):
            concept, next_id, multiple_concepts = get_label('Concept', label, multiple_concepts, sent, next_id, word)
            if not concept:
                action, next_id, multiple_actions = get_label('Action', label, multiple_actions, sent, next_id, word)
                if not action:
                    reference, next_id, multiple_references = get_label('Reference', label, multiple_references, sent, next_id, word)
                    if not reference:
                        _, next_id, multiple_predicates = get_label('Predicate', label, multiple_predicates, sent, next_id, word)
        next_id = create_keyphrase(sent, 'Concept', next_id, multiple_concepts)
        next_id = create_keyphrase(sent, 'Action', next_id, multiple_actions)
        next_id = create_keyphrase(sent, 'Predicate', next_id, multiple_predicates)
        next_id = create_keyphrase(sent, 'Reference', next_id, multiple_references)


def create_keyphrase(sent, label, next_id, multiple):
    if not multiple: 
        return next_id
    sent.keyphrases.append(Keyphrase(sent, label, next_id, multiple))
    return next_id + 1   


def get_label(label, pred_label, multiple, sent, next_id, word):
    if label not in pred_label:
        return False, next_id, multiple
    if pred_label == 'B-' + label:
        next_id = create_keyphrase(sent, label, next_id, multiple)
        multiple = []
    try:
        idx = multiple[-1][-1]
    except IndexError:
        idx = 0
    i = sent.text.index(word.text, idx)
    span = i, i + len(word)
    multiple.append(span)
    return True, next_id, multiple