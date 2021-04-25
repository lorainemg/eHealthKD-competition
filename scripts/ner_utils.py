import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spacy
import networkx as nx
from anntools import Sentence, Keyphrase, Collection
from utils import find_match, detect_language, nlp_es, nlp_en
from itertools import chain, zip_longest
import numpy as np
import re

tag_re = re.compile(r'([BILUOV])-(\w+)')
sufixes = tuple(nlp_es.Defaults.suffixes) + (r'%', r'\.')
suffix_re = spacy.util.compile_suffix_regex(sufixes)
nlp_es.tokenizer.suffix_search = suffix_re.search


################################ Preprocessing ################################
# Preprocess the features, getting the training instances (features + labels)
def select_tag(labels):
    if not labels:
        return "O"
    if len(labels) == 1:
        return labels[0]
    entity = tag_re.match(labels[0]).group(2)
    tags = [tag[0] for tag in labels]
    tag = "U" if ("U" in tags and "B" not in tags and "L" not in tags) else "V"
    return f'{tag}-{entity}'


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


def get_labels(tokens, sentence: Sentence):
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
        labels = find_match(idx, idx + n, sentence.keyphrases)
        tag = select_tag(labels)
        instances[(idx, idx + n)] = tag
    return instances.values()


def load_training_entities(sentence: Sentence, char2idx):
    lang = detect_language(sentence.text)
    nlp = nlp_es if lang == 'es' else nlp_en
    doc = nlp(sentence.text)
    features, X_char = get_features(doc, char2idx)
    labels = get_labels(doc, sentence)
    return features, X_char, list(labels)


def load_testing_entities(sentence: Sentence, char2idx):
    lang = detect_language(sentence.text)
    nlp = nlp_es if lang == 'es' else nlp_en
    doc = nlp(sentence.text)
    return get_features(doc, char2idx)


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
    if pred_label in ['B-' + label, 'U-' + label]:
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


# ---------------------------------------- #
def postprocessing_labels1(labels, indices, sentences, classes):
    next_id = 0
    for sent_labels, index in zip(labels, indices):
        sent = sentences[index]
        lang = detect_language(sent.text)
        tokens = nlp_en.tokenizer(sent.text) if lang == 'en' else nlp_es.tokenizer(sent.text)
        next_id = make_sentence(tokens, sent_labels, classes, sent, next_id)


def make_sentence(tokens, labels, classes, sentence, last_idx):
    entities = set(l[2:] for l in classes if l != 'O')
    for entity in entities:
        bilouv = []
        for tag in labels:
            if tag.endswith(entity):
                bilouv.append(tag[0])
            else:
                bilouv.append('O')

        spans = from_bilouv(bilouv, tokens)
        sentence.keyphrases.extend(Keyphrase(sentence, entity, last_idx + i, sp) for i, sp in enumerate(spans))
        last_idx += len(spans)
    return last_idx


def from_bilouv(bilouv, tokens):
    entities = [x for x in discontinuous_match(bilouv, tokens)]
    for i, (tag, word) in enumerate(zip(bilouv, tokens)):
        if tag == 'U':
            entities.append([word])
            bilouv[i] = 'O'
        elif tag == 'V':
            bilouv[i] = 'I'

    multiple = []
    for label, word in zip(bilouv, tokens):
        if label == 'L':
            entities.append(multiple + [word])
            multiple = []
        if label == 'B':
            if multiple:
                entities.append(multiple)
                multiple = []
            multiple.append(word)
        elif label != 'O':
            multiple.append(word)
    if len(multiple) > 0:
        entities.append(multiple)
    return [[(tok.idx, tok.idx + len(tok)) for tok in tokens] for tokens in entities]


def discontinuous_match(bilouv, tokens):
    entities = []
    for i, tag in enumerate(bilouv):
        if tag != "V":
            continue
        for entity_ids in _full_overlap(bilouv, list(range(len(tokens))), i):
            entity = []
            for idx in entity_ids:
                entity.append(tokens[idx])
                bilouv[idx] = "O"
            entities.append(entity)
    return entities


def _full_overlap(bilouv, sentence, index):
    left = _right_to_left_overlap(bilouv[:index + 1], sentence[:index + 1])
    right = _left_to_right_overlap(bilouv[index:], sentence[index:])

    full = []
    for l, r in zip_longest(left, right, fillvalue=[]):
        new = l + r[1:] if len(l) > len(r) else l[:-1] + r
        full.append(new)
    return full


def _left_to_right_overlap(biluov, sentence):
    return _build_overlap(biluov, sentence, "L")


def _right_to_left_overlap(biluov, sentence):
    inverse = _build_overlap(reversed(biluov), reversed(sentence), "B")
    for x in inverse:
        x.reverse()
    return inverse


def _build_overlap(biluov, sentence, finisher):
    prefix = []
    complete = []

    one_shot = zip(biluov, sentence)
    tag, word = next(one_shot)

    try:
        while tag in ("V", "O"):
            if tag == "V":
                prefix.append(word)
            tag, word = next(one_shot)

        on_build = []
        while tag in ("O", "I", "U", finisher):
            if tag == "I":
                on_build.append(word)
            elif tag == finisher:
                complete.append(prefix + on_build + [word])
                on_build = []
            elif tag == "U":
                complete.append([word])
            tag, word = next(one_shot)
    except StopIteration:
        pass

    if len(complete) == 1:
        complete.append(prefix)

    return complete
