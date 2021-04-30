from typing import List
from anntools import Relation, Sentence, Collection
from utils import find_keyphrase_by_span, nlp_es, nlp_en, detect_language, get_dependency_graph, lowest_common_ancestor
from utils import get_dependency_path
from itertools import chain
import random
import os
import fasttext
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def get_keyphrases_pairs(keyphrases):
    """Makes keyphrases pairs to extract the relation"""
    return [(k1, k2) for k1 in keyphrases for k2 in keyphrases]


################################ Preprocessing ################################
# Preprocess the features, getting the training instances (features + labels)
def find_keyphrase_tokens(sentence: Sentence, doc: List):
    """Returns the spacy tokens corresponding to a keyphrase"""
    text = sentence.text
    keyphrases = {}
    i = 0
    for token in doc:
        idx = text.index(token.text, i)
        n = len(token.text)
        i = idx + n
        keyphrases_ids = find_keyphrase_by_span(idx, idx + n, sentence.keyphrases)
        for keyphrase_id in keyphrases_ids:
            try:
                keyphrases[keyphrase_id].append(token)
            except KeyError:
                keyphrases[keyphrase_id] = [token]
    return keyphrases


def get_features(keyphrase1, keyphrase2, token1, token2, path_len):
    return {
        'origin_dtag': token1.dep_,
        'origin_pos': token1.pos_,
        'destination_dtag': token2.dep_,
        'destination_pos': token2.pos_,
        # 'origin': token1.lemma_,
        # 'destination': token2.lemma_,
        'origin_tag': keyphrase1.label,
        'destination_tag': keyphrase2.label,
        'path_len': path_len
    }


def get_dependency_feat(tokens, keyphrases, k1, k2, digraph, graph):
    lca1 = lowest_common_ancestor(keyphrases[k1.id], digraph)
    lca2 = lowest_common_ancestor(keyphrases[k2.id], digraph)
    token1 = tokens[lca1]
    token2 = tokens[lca2]
    dep_path, dep_len = get_dependency_path(graph, token1.i, token2.i, tokens)
    return token1, token2, dep_path, dep_len


def get_vec(token1, token2, model):
        
    r = list(model.get_word_vector(token1.text))
    r1 = list(model.get_word_vector(token2.text))
    
    return r + r1


def load_training_relations(sentence: Sentence, model, negative_sampling=1.0):
    lang = detect_language(sentence.text)
    nlp = nlp_es if lang == 'es' else nlp_en
    tokens = nlp(sentence.text)

    features = []
    labels = []
    feat_path = []
    my_embedding = []

    keyphrases = find_keyphrase_tokens(sentence, tokens)
    graph = get_dependency_graph(tokens, directed=False)
    digraph = get_dependency_graph(tokens, directed=True)

    for relation in sentence.relations:
        destiny = relation.to_phrase
        origin = relation.from_phrase

        token1, token2, dep_path, dep_len = get_dependency_feat(tokens, keyphrases, origin, destiny, digraph, graph)
        path, path_len = get_dependency_path(graph, token1.i, token2.i, tokens)
        features.append(get_features(origin, destiny, token1, token2, dep_len))
        feat_path.append(path)
        labels.append(relation.label)
        my_embedding.append(get_vec(token1, token2, model))

    for k1 in sentence.keyphrases:
        for k2 in sentence.keyphrases:
            if not sentence.find_relations(k1, k2) and random.uniform(0, 1) < negative_sampling:
                t1, t2, dep_path, dep_len = get_dependency_feat(tokens, keyphrases, k1, k2, digraph, graph)
                path, path_len = get_dependency_path(graph, t1.i, t2.i, tokens)
                features.append(get_features(k1, k2, t1, t2, path_len))
                feat_path.append(path)
                labels.append("empty")
                my_embedding.append(get_vec(t1, t2, model))

    return features, feat_path, labels , my_embedding


def load_testing_relations(sentence: Sentence, model):
    lang = detect_language(sentence.text)
    nlp = nlp_es if lang == 'es' else nlp_en
    tokens = nlp(sentence.text)

    keyphrases = find_keyphrase_tokens(sentence, tokens)
    graph = get_dependency_graph(tokens, directed=False)
    digraph = get_dependency_graph(tokens, directed=True)

    features = []
    feat_path = []
    my_embedding = []
    for k1, k2 in get_keyphrases_pairs(sentence.keyphrases):
        t1, t2, dep_path, dep_len = get_dependency_feat(tokens, keyphrases, k1, k2, digraph, graph)
        path, path_len = get_dependency_path(graph, t1.i, t2.i, tokens)
        features.append(get_features(k1, k2, t1, t2, path_len))
        feat_path.append(path)
        my_embedding.append(get_vec(t1, t2, model))
    return features, feat_path, my_embedding


def train_by_shape(X, X_dep_feat, y, my_embedding):
    """
    Separates the features and labels by its shape
    :param X: Word-features
    :param y: Labels
    :return: 3 dictionaries of sublists of the parameters separated by there size
    """
    x_shapes = {}
    y_shapes = {}
    x_dep_shapes = {}
    my_embedding_shapes = {}
    for itemX, itemXDep, itemY, itemZ in zip(X, X_dep_feat, y, my_embedding):
        try:
            x_shapes[itemX.shape[0]].append(itemX)
            y_shapes[itemX.shape[0]].append(itemY)
            x_dep_shapes[itemX.shape[0]].append(itemXDep)
            my_embedding_shapes[itemX.shape[0]].append(itemZ)
        except KeyError:
            x_shapes[itemX.shape[0]] = [itemX]  # initially a list, because we're going to append items
            y_shapes[itemX.shape[0]] = [itemY]
            x_dep_shapes[itemX.shape[0]] = [itemXDep]
            my_embedding_shapes[itemX.shape[0]]= [itemZ]
    return x_shapes, x_dep_shapes,my_embedding_shapes, y_shapes


def predict_by_shape(X, X_dep_feat, my_embedding):
    """
    Separates the features by its shape
    :param X: Word-features
    :return: 2 dictionaries of sublists of the parameters separated by there size
    """
    x_shapes = {}
    indices = {}
    x_dep_shapes = {}
    my_embedding_shapes = {}
    for i, (itemX, itemXDep, itemZ) in enumerate(zip(X, X_dep_feat, my_embedding)):
        try:
            x_shapes[len(itemX)].append(itemX)
            indices[len(itemX)].append(i)
            x_dep_shapes[itemX.shape[0]].append(itemXDep)
            my_embedding_shapes[len(itemX)].append(itemZ)
        except KeyError:
            x_shapes[len(itemX)] = [itemX]  # initially a list, because we're going to append items
            indices[len(itemX)] = [i]
            x_dep_shapes[itemX.shape[0]] = [itemXDep]
            my_embedding_shapes[len(itemX)] =[itemZ]
    return x_shapes.values(), x_dep_shapes.values(), my_embedding_shapes.values(), chain(*indices.values())


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
