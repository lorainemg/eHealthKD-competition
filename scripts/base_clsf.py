from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import metrics
import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt


class BaseClassifier:
    """Base classifier of the ner and re task"""

    def __init__(self):
        self.model = None
        self.max_len = 50
        # DictVectorizer is used to convert the features
        self.vectorizer = DictVectorizer()
        # Label Encoder is used to transform the labels
        # Label encoder transforms labels in strings as numbers
        self.encoder = LabelEncoder()
        # Weights of all the labels, we are in presence of an imbalance class problem
        self.weights = None
        # different metrics to take into account
        self.metrics = [
            metrics.TruePositives(name='tp'),
            metrics.FalsePositives(name='fp'),
            metrics.TrueNegatives(name='tn'),
            metrics.FalseNegatives(name='fn'),
            metrics.BinaryAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            metrics.AUC(name='auc'),
            metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
        ]

    def preprocess_features(self, features, train=True):
        """
        The features are converted to vectors and their shape is adjusted
        """
        # X = self._padding_dicts(features, null_value)
        X = features
        if train:
            # first the vectorizer must fit the examples
            self.vectorizer.fit(list(itertools.chain(*X)))
        # after all the examples are transformed
        X = [self.vectorizer.transform(sent).todense() for sent in X]
        self.n_features = X[0].shape[-1]
        #         X = X.reshape(1921, 15, X.shape[1])
        return X

    def preprocess_labels(self, labels):
        """
        The labels are converted in vectors and their shape is adjusted.
        """
        # As with DictVectorizer, all the labels are fit a\nd transformed
        self.encoder.fit(list(itertools.chain(*labels)))
        y = [self.encoder.transform(label) for label in labels]
        # the padding is done
        # y = pad_sequences(maxlen=self.max_len, sequences=y, padding="post", value=self.encoder.transform([null_value])[0])
        #         y = y.reshape(1921, 15, y.shape[1])
        # the labels are one-hot encoded, i.e, the number are represented in arrays.
        y = [to_categorical(elem, num_classes=len(self.encoder.classes_)) for elem in y]
        self.n_labels = y[0].shape[-1]
        return y

    def get_weights(self, labels):
        unique_classes = np.array(self.encoder.classes_)
        labels = np.concatenate(labels)
        self.weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
        # self.weights = {i: v for i, v in enumerate(weights)}

    # def _padding_dicts(self, X):
    #     '''
    #     Auxiliar function because the keras.pad_sequences does not accept dictionaries.
    #     '''
    #     new_X = []
    #     for seq in X:
    #         new_seq = []
    #         for i in range(self.max_len):
    #             try:
    #                 new_seq.append(seq[i])
    #             except:
    #                 new_seq.append(null_value)
    #         new_X.append(new_seq)
    #     return new_X
    def fit_model(self, X, y, plot=False):
        raise NotImplementedError()

    def save_model(self, name):
        self.model.save(fr'resources/{name}_model.h5')
        pickle.dump(self.vectorizer, open(fr'resources/{name}_vectorizer.pkl', 'wb'))
        pickle.dump(self.encoder, open(fr'resources/{name}_encoder.pkl', 'wb'))

    def load_model(self, name):
        self.model = load_model(fr'resources/{name}_model.h5')
        self.vectorizer = pickle.load(open(fr"resources/{name}_vectorizer.pkl", 'rb'))
        self.encoder = pickle.load(open(fr"resources/{name}_encoder.pkl", 'rb'))

    def convert_to_label(self, pred):
        """Converts the predictions of the model in strings labels"""
        out = []
        for pred_i in pred:
            out_i = []
            for p in pred_i:
                p_i = np.argmax(p)
                out_i.append(self.encoder.inverse_transform([p_i])[0])
            out.append(out_i)
        return out
