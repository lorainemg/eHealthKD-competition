from typing import List
from anntools import Keyphrase
from tensorflow.keras.utils import Sequence
import numpy as np
from itertools import groupby, chain
from tensorflow.keras.callbacks import Callback
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
# These are the imports which we need in the utils file
from tensorflow.keras.metrics import Metric
import tensorflow as tf
import tensorflow.keras.backend as K
import networkx as nx

import fasttext
import re

import es_core_news_sm, en_core_web_sm

nlp_es = es_core_news_sm.load()
nlp_en = en_core_web_sm.load()

# model for detecting the language of a sentence
lid_model = fasttext.load_model("resources/lid.176.ftz")
# regex to parse the output of the language identification model
lid_regex = re.compile(r"__label__(en|es)")


def detect_language(sentence: str) -> str:
    # The tokens and its features are extracted with spacy
    lang, _ = lid_model.predict(sentence)
    try:
        lang = lid_regex.findall(lang[0])[0]
    except IndexError:
        lang = 'es'
    return lang


def match_inner(x, y, i, j):
    """
    Match whether the token is inside the entity
    :param x: The start of the entity
    :param y: The end of the entity
    :param i: The start of the token
    :param j: The end of the token
    """
    return x <= i and y >= j


def match_outside(x, y, i, j):
    """
    Match whether the token is outside the entity
    :param x: The start of the entity
    :param y: The end of the entity
    :param i: The start of the token
    :param j: The end of the token
    """
    return i <= x and j >= y


def find_keyphrase_by_span(start: int, end: int, keyphrases: List[Keyphrase]):
    keyphrases_id = []
    for keyphrase in keyphrases:
        for x, y in keyphrase.spans:
            # if the token is contained in the keyphrase or vice versa
            if match_inner(x, y, start, end) or match_outside(x, y, start, end):
                keyphrases_id.append(keyphrase.id)
                break
    return keyphrases_id


def find_match(start: int, end: int, keyphrases: List[Keyphrase]):
    """
    Returns the keyphrase id and the tag of a keyphrase based on the indices that a
    token occupies in a given sentence
    """
    labels = []
    for keyphrase in keyphrases:
        for idx, (x, y) in enumerate(keyphrase.spans):
            end_span = len(keyphrase.spans) - 1
            # if the token is contained in the keyphrase
            if match_inner(x, y, start, end) or match_outside(x, y, start, end):
                if len(keyphrase.spans) == 1:
                    labels.append('U-' + keyphrase.label)
                elif idx == 0:
                    labels.append('B-' + keyphrase.label)
                elif idx == end_span:
                    labels.append('L-' + keyphrase.label)
                else:
                    labels.append('I-' + keyphrase.label)
                break
    return labels


def get_dependency_graph(tokens: List, directed=False):
    """Constructs the dependency graph, directed or undirected"""
    graph = nx.DiGraph()
    for token in tokens:
        for child in token.children:
            if directed:
                graph.add_edge(token.i, child.i)
            else:
                graph.add_edge(token.i, child.i, attr_dict={'dir': '/'})
                graph.add_edge(child.i, token.i, attr_dict={'dir': '\\'})
    return graph


def lowest_common_ancestor(tokens: List, graph: nx.DiGraph) -> int:
    """Given a list of tokens, it finds his lowest common ancestor in the dependency graph"""
    if len(tokens) == 1:
        return tokens[0].i
    else:
        try:
            lca = nx.lowest_common_ancestor(graph, tokens[0].i, tokens[1].i)
        except nx.exception.NodeNotFound:
            lca = min(tokens[0].i, tokens[1].i)
        for token in tokens[2:]:
            try:
                lca = nx.lowest_common_ancestor(graph, lca, token.i)
            except nx.exception.NodeNotFound:
                lca = min(token.i, lca)
        return lca if lca is not None else tokens[0].i


def get_dependency_path(graph: nx.DiGraph, token1: int, token2: int, sent):
    """
    Gets the shortest path between the two tokens in the dependency tree.
    Returns the path and its length, otherwise returns null
    """
    path = []
    try:
        path_list = nx.dijkstra_path(graph, token1, token2)
        prev_node = token1
        for node in path_list:
            # try:
            #     dir = graph[prev_node][node]['attr_dict']['dir']
            # except:
            #     dir = 'null'
            path.append(sent[node].dep_)
            prev_node = node
        return path, len(path_list)
    except:
        return path, 'null'


class MyBatchGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, X, y, batch_size=1, shuffle=True):
        """Initialization"""
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        """Shuffles indexes after each epoch"""
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        yb = np.empty((self.batch_size, *self.y[index].shape))
        # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            yb[s] = self.y[index]
        return Xb, yb


def weighted_loss(originalLossFunc, weightsList):
    def loss_func(true, pred):
        axis = -1  # if channels last
        # axis=  1 #if channels first

        # argmax returns the index of the element with the greatest value
        # done in the class axis, it returns the class index
        classSelectors = K.argmax(true, axis=axis)

        # if your loss is sparse, use only true as classSelectors
        tf.cast(classSelectors, tf.int64)
        # classSelectors = classSelectors.astype(np.int32)
        # print(type(classSelectors))

        # considering weights are ordered by class, for each class
        # true(1) if the class index is equal to the weight index
        classSelectors = [K.equal(np.int64(i), classSelectors) for i in range(len(weightsList))]

        # casting boolean to float for calculations
        # each tensor in the list contains 1 where ground true class is equal to its index
        # if you sum all these, you will get a tensor full of ones.
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        # for each of the selections above, multiply their respective weight
        weights = [sel * w for sel, w in zip(classSelectors, weightsList)]

        # sums all the selections
        # result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]

        # make sure your originalLossFunc only collapses the class axis
        # you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true, pred)
        loss = loss * weightMultiplier

        return loss

    return loss_func
