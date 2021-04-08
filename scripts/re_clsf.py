from anntools import Collection, Keyphrase, Relation

from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Input, Embedding, Lambda
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from re_preprocessing import get_instances

class REClassifier:
    "Classifier for the relation extraction task"
    def __init__(self):
        self.n_timesteps = 15
        self.model = None

    def train(self, collection:Collection):
        features, labels = self.get_sentences(collection)
        X, y = self.preprocessing(features, labels)
        self.get_bi_lstm_model('concat')
        return self.fit_model(X, y, True)


    def get_bi_lstm_model(self, mode:str):
        inputs = Input(shape=(self.X_shape[1], self.X_shape[2]))
#         outputs = Embedding(input_dim=35179, output_dim=20,
#                           input_length=self.X_shape[1], mask_zero=True)(inputs)  # 20-dim embedding
        outputs = Bidirectional(LSTM(units=512, return_sequences=True,
                                   recurrent_dropout=0.1))(inputs)  # variational biLSTM
        outputs = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(outputs)
        outputs = TimeDistributed(Dense(self.y_shape[2], activation="softmax"))(outputs)  # a dense layer as suggested by neuralNer
#         crf = CRF(8)  # CRF layer
#         out = crf(outputs)  # output

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#         model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        # model.compile(loss='binary_crossentropy', optimizer='adam')
        self.model = model

    def preprocessing(self, features, labels):
        self.max_len = 50
        
        vectorizer = DictVectorizer()
        X = self._padding_dicts(features)
        [vectorizer.fit(sent) for sent in X]
        X = np.array([vectorizer.transform(sent).todense() for sent in X])
#         X = X.reshape(1921, 15, X.shape[1])
        self.X_shape = X.shape
    
        encoder = LabelEncoder()
#         print(labels)
        y = [encoder.fit_transform(label) for label in labels]
        y = pad_sequences(maxlen=50, sequences=y, padding="post", value=encoder.transform(['empty'])[0])
#         y = y.reshape(1921, 15, y.shape[1])    
        y = to_categorical(y)
        self.y_shape = y.shape
        return X, y

    def _padding_dicts(self, X):
        new_X = []
        for seq in X:
            new_seq = []
            for i in range(self.max_len):
                try:
                    new_seq.append(seq[i])
                except:
                    new_seq.append({'dep': 0, 'pos': 0})
            new_X.append(new_seq)
        return new_X
            
    def fit_model(self, X, y, plot=False):
        hist = self.model.fit(X, y, batch_size=32, epochs=5,
                    validation_split=0.2, verbose=1)
        if plot:
            plt.style.use("ggplot")
            plt.figure(figsize=(12, 12))
            plt.plot(hist["acc"])
            plt.plot(hist["val_acc"])
            plt.show() 

    def get_sentences(self, collection:Collection):
        features = []
        labels = []
#         self.max_len = 0
        for sentence in collection:
            feat, label = get_instances(sentence)
#             self.max_len = max(self.max_len, len(feat))
            features.append(feat)
            labels.append(label)
        return features, labels
    
    def run(self, collection: Collection):
        collection = collection.clone()
        # returns a collection with everything annotated
        return collection

if __name__ == "__main__":
    collection = Collection().load_dir(Path('2021/ref/training'))
    ner_clf = REClassifier()
    ner_clf.train(collection)