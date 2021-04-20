import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spacy
import networkx as nx
from anntools import Sentence, Keyphrase, Collection
from utils import find_keyphrase_by_span, detect_language, nlp_es, nlp_en
from itertools import chain
import numpy as np

sufixes = tuple(nlp_es.Defaults.suffixes) + (r'%', r'\.')
suffix_re = spacy.util.compile_suffix_regex(sufixes)
nlp_es.tokenizer.suffix_search = suffix_re.search


################################ Preprocessing ################################
# Preprocess the features, getting the training instances (features + labels)
def get_features(tokens, char2idx):
    """
    Given a list of tokens returns the features of those tokens.
    They will be used for training in the first task (Name Entity Recognition)
    """
    graph = nx.DiGraph()
    digraph = nx.DiGraph()
    features = []
    X_char = []
    for token in tokens:
        word_seq = []
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
        for i in range(10):
            try:
                w_i = token.text[i]
                word_seq.append(char2idx[w_i])
            except KeyError:
                word_seq.append(char2idx['UNK'])
            except IndexError:
                word_seq.append(char2idx['PAD'])
        X_char.append(np.array(word_seq))
    return features, X_char


def get_labels(tokens, sentence: Sentence, nlp):
    """
    Given a list of tokens and the sentences containing them returns the labels of those tokens.
    They will be used for training in the first task (Name Entity Recognition)
    """
    text = sentence.text
    instances = {}
    i = 0
    for token in tokens:
        idx = text.index(token.text, i)
        n = len(token.text)
        i = idx + n
        # instances[(idx, idx + n)] = classify_as_keyphrase(token.text, sentence.keyphrases, idx)
        _, instances[(idx, idx + n)] = find_keyphrase_by_span(idx, idx + n, sentence.keyphrases, sentence.text, nlp)
    return instances.values()


def get_instances(sentence: Sentence, char2idx, labels=True):
    """
    Makes all the analysis of the sentence according to spacy.
    Returns the features and the labels corresponding to those features in the sentence.
    """
    lang = detect_language(sentence.text)
    nlp = nlp_es if lang == 'es' else nlp_en
    doc = nlp(sentence.text)
    features, X_char = get_features(doc, char2idx)
    if labels:
        labels = get_labels(doc, sentence, nlp)
        return features, X_char, list(labels)
    else:
        return features, X_char


def get_char2idx(collection: Collection):
    """
    Gets the char dicctionary
    :param collection: Collection with all the sentences
    :return: Dictionary with all the characters in the collection
    """
    chars = set([w_i for sentence in collection.sentences for w_i in sentence.text])
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx['PAD'] = 0
    char2idx['UNK'] = 1
    return char2idx


def train_by_shape(X, y, X_char):
    """
    Separates the features and labels by its shape
    :param X: Word-features
    :param y: Labels
    :param X_char: X-char mappings
    :return: 3 dictionaries of sublists of the parameters separated by there size
    """
    x_shapes = {}
    y_shapes = {}
    x_char_shapes = {}
    for itemX, X_char, itemY in zip(X, X_char, y):
        try:
            x_shapes[itemX.shape[0]].append(itemX)
            x_char_shapes[itemX.shape[0]].append(X_char)
            y_shapes[itemX.shape[0]].append(itemY)
        except:
            x_shapes[itemX.shape[0]] = [itemX]  # initially a list, because we're going to append items
            x_char_shapes[itemX.shape[0]] = [X_char]
            y_shapes[itemX.shape[0]] = [itemY]
    return x_shapes, x_char_shapes, y_shapes


def predict_by_shape(X, X_char):
    """
    Separates the features by its shape
    :param X: Word-features
    :param X_char: X-char mappings
    :return: 2 dictionaries of sublists of the parameters separated by there size
    """
    x_char_shapes = {}
    x_shapes = {}
    indices = {}
    for i, (itemX, X_char) in enumerate(zip(X, X_char)):
        try:
            x_char_shapes[itemX.shape[0]].append(X_char)
            x_shapes[len(itemX)].append(itemX)
            indices[len(itemX)].append(i)
        except:
            x_shapes[len(itemX)] = [itemX]  # initially a list, because we're going to append items
            x_char_shapes[itemX.shape[0]] = [X_char]
            indices[len(itemX)] = [i]
    return x_shapes.values(), x_char_shapes.values(), chain(*indices.values())


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
        lang = detect_language(sent.text)
        tokens = nlp_en.tokenizer(sent.text) if lang == 'en' else nlp_es.tokenizer(sent.text)
        for label, word in zip(sent_label, tokens):
            concept, next_id, multiple_concepts = get_label('Concept', label, multiple_concepts, sent, next_id, word)
            if not concept:
                action, next_id, multiple_actions = get_label('Action', label, multiple_actions, sent, next_id, word)
                if not action:
                    reference, next_id, multiple_references = get_label('Reference', label, multiple_references, sent,
                                                                        next_id, word)
                    if not reference:
                        _, next_id, multiple_predicates = get_label('Predicate', label, multiple_predicates, sent,
                                                                    next_id, word)
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
