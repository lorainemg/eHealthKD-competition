from typing import List
from anntools import Relation, Sentence, Collection
from utils import find_keyphrase_by_span, nlp_es, nlp_en, detect_language, get_dependency_graph, lowest_common_ancestor
from utils import get_dependency_path
from itertools import chain
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_keyphrases_pairs(keyphrases):
    """Makes keyphrases pairs to extract the relation"""
    return [(k1, k2) for k1 in keyphrases for k2 in keyphrases]


################################ Preprocessing ################################
# Preprocess the features, getting the training instances (features + labels)
def find_keyphrase_tokens(sentence: Sentence, doc: List, nlp):
    """Returns the spacy tokens corresponding to a keyphrase"""
    text = sentence.text
    keyphrases = {}
    i = 0
    for token in doc:
        idx = text.index(token.text, i)
        n = len(token.text)
        i = idx + n
        keyphrases_ids, _ = find_keyphrase_by_span(idx, idx + n, sentence.keyphrases, text, nlp)
        if keyphrases_ids is None:
            continue
        for keyphrase_id in keyphrases_ids:
            try:
                keyphrases[keyphrase_id].append(token)
            except KeyError:
                keyphrases[keyphrase_id] = [token]
    return keyphrases


def get_features(tokens, keyphrase1, keyphrase2, keyphrases, graph):
    lca1 = lowest_common_ancestor(keyphrases[keyphrase1.id], graph)
    lca2 = lowest_common_ancestor(keyphrases[keyphrase2.id], graph)
    token1 = tokens[lca1]
    token2 = tokens[lca2]
    dep_path, dep_len = get_dependency_path(graph, token1.i, token2.i, tokens)
    return {
        'origin_dtag': token1.dep_,
        'origin_pos': token1.pos_,
        'destination_dtag': token2.dep_,
        'destination_pos': token2.pos_,
        # 'origin': token1.lemma_,
        # 'destination': token2.lemma_,
        'origin_tag': keyphrase1.label,
        'destination_tag': keyphrase2.label,
        'dep_path': dep_path,
        'dep_len': dep_len
    }


def load_training_relations(sentence: Sentence, negative_sampling=1.0):
    lang = detect_language(sentence.text)
    nlp = nlp_es if lang == 'es' else nlp_en
    tokens = nlp(sentence.text)

    features = []
    labels = []

    keyphrases = find_keyphrase_tokens(sentence, tokens, nlp)
    graph = get_dependency_graph(tokens, directed=True)

    for relation in sentence.relations:
        destiny = relation.to_phrase
        origin = relation.from_phrase

        features.append(get_features(tokens, origin, destiny, keyphrases, graph))
        labels.append(relation.label)

    for k1 in sentence.keyphrases:
        for k2 in sentence.keyphrases:
            if not sentence.find_relations(k1, k2) and random.uniform(0, 1) < negative_sampling:
                features.append(get_features(tokens, k1, k2, keyphrases, graph))
                labels.append("empty")
    return features, labels


def load_testing_relations(sentence: Sentence):
    lang = detect_language(sentence.text)
    nlp = nlp_es if lang == 'es' else nlp_en
    tokens = nlp(sentence.text)

    keyphrases = find_keyphrase_tokens(sentence, tokens, nlp)
    graph = get_dependency_graph(tokens, directed=True)

    features = []
    for k1, k2 in get_keyphrases_pairs(sentence.keyphrases):
        features.append(get_features(tokens, k1, k2, keyphrases, graph))
    return features


def train_by_shape(X, y):
    """
    Separates the features and labels by its shape
    :param X: Word-features
    :param y: Labels
    :return: 3 dictionaries of sublists of the parameters separated by there size
    """
    x_shapes = {}
    y_shapes = {}
    for itemX, itemY in zip(X, y):
        try:
            x_shapes[itemX.shape[0]].append(itemX)
            y_shapes[itemX.shape[0]].append(itemY)
        except:
            x_shapes[itemX.shape[0]] = [itemX]  # initially a list, because we're going to append items
            y_shapes[itemX.shape[0]] = [itemY]
    return x_shapes, y_shapes


def predict_by_shape(X):
    """
    Separates the features by its shape
    :param X: Word-features
    :return: 2 dictionaries of sublists of the parameters separated by there size
    """
    x_shapes = {}
    indices = {}
    for i, itemX in enumerate(X):
        try:
            x_shapes[len(itemX)].append(itemX)
            indices[len(itemX)].append(i)
        except:
            x_shapes[len(itemX)] = [itemX]  # initially a list, because we're going to append items
            indices[len(itemX)] = [i]
    return x_shapes.values(), chain(*indices.values())


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
