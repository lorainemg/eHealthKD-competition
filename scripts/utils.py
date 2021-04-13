from typing import List
from anntools import Keyphrase
from tensorflow.keras.utils import Sequence
import numpy as np
from itertools import groupby, chain

def find_keyphrase_by_span(i:int, j:int, keyphrases:List[Keyphrase], sentence:str, nlp):
    '''
    Returns the keyphrase id and the tag of a keyphrase based on the indices that a 
    token occupies in a given sentence
    '''
    # TODO: This has to handle correctly multitokens to work properly, so it should return a list of id / labels
    token = sentence[i:j]
    for keyphrase in keyphrases:
        for idx, (x, y) in enumerate(keyphrase.spans):
            word = sentence[x:y]
            if x <= i and y >= j:
                if idx == 0: return keyphrase.id, 'B-' + keyphrase.label
                else: return keyphrase.id, 'I-' + keyphrase.label 
            elif i <= x and j >= y:
                if idx == 0:
                    tokens = nlp.tokenizer(word)
                    if i + len(tokens[0]) <= j:
                        return keyphrase.id, 'B-' + keyphrase.label
                return keyphrase.id, 'I-' + keyphrase.label   
    return None, 'O'

def train_by_shape(X, y):
    x_shapes = {}
    y_shapes = {}
    for itemX,itemY in zip(X,y):
        try:
            x_shapes[itemX.shape[0]].append(itemX)
            y_shapes[itemX.shape[0]].append(itemY)
        except:
            x_shapes[itemX.shape[0]] = [itemX] #initially a list, because we're going to append items
            y_shapes[itemX.shape[0]] = [itemY]
    return x_shapes, y_shapes

def predict_by_shape(X):
#     return [list(g) for k, g in groupby(X, len)]
    x_shapes = {}
    indeces = {}
    for i, itemX in enumerate(X):
        try:
            x_shapes[len(itemX)].append(itemX)
            indeces[len(itemX)].append(i)
        except:
            x_shapes[len(itemX)] = [itemX] #initially a list, because we're going to append items
            indeces[len(itemX)] = [i]
    return x_shapes.values(), chain(*indeces.values())

class MyBatchGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=1, shuffle=True):
        'Initialization'
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
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