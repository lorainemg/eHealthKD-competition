import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spacy
import networkx as nx
# import es_core_news_sm

nlp = spacy.load('es_core_news_sm')

sufixes = nlp.Defaults.suffixes + [r'%',r'\.']
suffix_re = spacy.util.compile_suffix_regex(sufixes)
nlp.tokenizer.suffix_search = suffix_re.search


def get_features(tokens):
    graph = nx.DiGraph()
    digraph = nx.DiGraph()
    features = []
    for token in tokens:
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

def classify_as_keyphrase(token:str, keyphrases, i:int, instances) -> str:
    n = len(token)
    for keyphrase in keyphrases:
        x,y = keyphrase.spans[0][0], keyphrase.spans[-1][1]
        if (x >= i and y <= i + n) or (i >= x and i+n <= y):
            word = next(keyphrase.tokens)
            # busca si es el principio de un keyprhase
            if word == token:
                return 'B-' + keyphrase.label
            # hay palabras que están mal tokenizadas, asi que se comprueba
            tokens = nlp.tokenizer(word)
            # se vuelve a tokenizar la palabra anotado, y si es el principio del token, entonces es el principio 
            if len(tokens) == 1 and token == tokens[0].text:
                return 'B-' + keyphrase.label
            for word, (x,y) in zip(keyphrase.tokens, keyphrase.spans):
                if (x >= i and y <= i + n) or (i >= x and i+n <= y):
                    if word == token:
                        return 'I-' + keyphrase.label
                    for tok in tokens:
                        if token == tok.text:
                            return 'I-' + keyphrase.label 
    return 'O'


def get_labels(tokens, sentence):
    text = sentence.text
    instances = {}
    i = 0
    for token in tokens:
        idx = text.index(token.text, i)
        n = len(token.text)
        i = idx + n 
        instances[(idx, idx + n)] = classify_as_keyphrase(token.text, sentence.keyphrases, idx, instances)
    return instances.values()


def get_instances(sentence):
    """
    Makes all the analysis of the sentence according to spacy
    """
    doc = nlp(sentence.text)
    features = get_features(doc)
    labels = get_labels(doc, sentence)
    return features, list(labels)

if __name__ == "__main__":
    f =  spacy_features("Hola esto es una prueba")
    print(f)