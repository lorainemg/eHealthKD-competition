import spacy
import networkx as nx

nlp = spacy.load('es_core_news_sm')

def spacy_features(sentence:str):
    """
    Makes all the analysis of the sentence according to spacy
    """
    doc = nlp(sentence)
    graph = nx.DiGraph()
    digraph = nx.DiGraph()
    features = {}
    for token in doc:
        for child in token.children:
            digraph.add_edge(token.i, child.i)
            graph.add_edge(token.i, child.i, {'dir': '/'})
            graph.add_edge(child.i, token.i, {'dir': '\\'})
            # edges.append((token.i, child.i))
        #? Dont know whats token.i
        if token.i in sentence:
            features[token.i] = {
                'head': token.head.i,
                'dep': token.dep_,
                'pos': token.pos_,
                'lemma': token.lemma_
            }
    features['undir-graph'] = graph #nx.Graph(edges)
    features['dir-graph'] = digraph #nx.DiGraph(edges)
    return features