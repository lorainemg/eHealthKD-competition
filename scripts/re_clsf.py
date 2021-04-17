from anntools import Collection

from pathlib import Path
# from typing import List

from base_clsf import BaseClassifier

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Input, Embedding, Lambda
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import metrics
from re_utils import get_instances, postprocessing_labels
from utils import predict_by_shape, Metrics, weighted_loss
import numpy as np
from tensorflow_addons.metrics import FBetaScore


class REClassifier(BaseClassifier):
    """Classifier for the relation extraction task"""

    def train(self, collection: Collection):
        """
        Wrapper function where of the process of training is done
        """
        features, labels = self.get_sentences(collection)
        X, y = self.preprocessing(features, labels)
        self.get_bi_lstm_model('concat')
        return self.fit_model(X, y)

    def get_bi_lstm_model(self, mode: str):
        """
        Construct the neural network architecture using the keras functional api.
        `mode` is the mode where the lstm are joined in the bidirectional layer, (its not currently being used)
        """
        inputs = Input(shape=(None, self.n_features))
        #         outputs = Embedding(input_dim=35179, output_dim=20,
        #                           input_length=self.X_shape[1], mask_zero=True)(inputs)  # 20-dim embedding
        outputs = Bidirectional(LSTM(units=32, return_sequences=True,
                                     recurrent_dropout=0.1))(inputs)  # variational biLSTM
        # outputs = Bidirectional(LSTM(units=512, return_sequences=True,
        #                    recurrent_dropout=0.2, dropout=0.2))(outputs)
        outputs = TimeDistributed(Dense(self.n_labels, activation="softmax"))(
            outputs)  # a dense layer as suggested by neuralNer
        #         crf = CRF(8)  # CRF layer
        #         out = crf(outputs)  # output

        METRICS = [
            metrics.TruePositives(name='tp'),
            metrics.FalsePositives(name='fp'),
            metrics.TrueNegatives(name='tn'),
            metrics.FalseNegatives(name='fn'),
            metrics.BinaryAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            metrics.AUC(name='auc'),
            metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", metrics=METRICS,
                      loss=weighted_loss(categorical_crossentropy, self.weights))
        #         model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        # model.compile(loss='binary_crossentropy', optimizer='adam')
        self.model = model

    def preprocessing(self, features, labels):
        """
        Handles the preprocessing step. The features and labels are converted in vectors \
            and their shape is adjusted.
        """
        X = self.preprocess_features(features)
        y = self.preprocess_labels(labels)
        self.weights = self.get_weights(labels)
        # self.X_shape = X.shape  
        # self.y_shape = y.shape
        return X, y

    def get_sentences(self, collection: Collection):
        """
        Giving a collection, the features and labels of its sentences are returned
        """
        features = []
        labels = []
        for sentence in collection:
            feat, label = get_instances(sentence, labels=True)
            features.append(feat)
            labels.append(label)
        return features, labels

    def get_features(self, collection: Collection):
        """Giving a collection, the features of its sentences are returned"""
        return [get_instances(sentence, labels=False) for sentence in collection]

    def test_model(self, collection: Collection):
        features = self.get_features(collection)
        X = self.preprocess_features(features, train=False)
        x_shapes, indices = predict_by_shape(X)
        pred = []
        for x_items in x_shapes:
            pred.extend(self.model.predict(np.asarray(x_items)))
        labels = self.convert_to_label(pred)
        print(labels)
        postprocessing_labels(labels, list(indices), collection)
        return collection.sentences

    def run(self, collection: Collection):
        collection = collection.clone()
        # returns a collection with everything annotated
        return collection


if __name__ == "__main__":
    collection = Collection().load_dir(Path('2021/ref/training'))
    dev_set = Collection().load(Path('2021/eval/develop/scenario1-main/output.txt'))
    ner_clf = REClassifier()
    ner_clf.train(collection)
    ner_clf.save_model('re')
    # ner_clf.load_model('re')
    ner_clf.test_model(dev_set)
