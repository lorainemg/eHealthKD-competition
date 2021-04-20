from anntools import Collection
from pathlib import Path

from ner_utils import get_instances, postprocessing_labels, get_char2idx, train_by_shape, predict_by_shape
from base_clsf import BaseClassifier
import score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Input, Embedding, concatenate, SpatialDropout1D
from tensorflow.keras.losses import categorical_crossentropy
# from utils import DataGeneratorPredict
from keras_crf import CRF
# from keras_crf import CRF
import numpy as np
import json


class NERClassifier(BaseClassifier):
    """Classifier for the name entity resolution task"""

    def train(self, collection: Collection):
        """
        Wrapper function where of the process of training is done
        """
        features, X_char,  labels = self.get_sentences(collection)
        X, y = self.preprocessing(features, labels)
        self.get_bi_lstm_model('concat')
        return self.fit_model((X, X_char), y)

    def get_bi_lstm_model(self, mode: str):
        """
        Construct the neural network architecture using the keras functional api.
        `mode` is the mode where the lstm are joined in the bidirectional layer, (its not currently being used)
        """
        # input for words
        inputs = Input(shape=(None, self.n_features))
        #         outputs = Embedding(input_dim=35179, output_dim=20,
        #                           input_length=self.X_shape[1], mask_zero=True)(inputs)  # 20-dim embedding


        # input for characters
        char_in = Input(shape=(None, 10,))
        emb_char = TimeDistributed(Embedding(input_dim=254, output_dim=10,
                                             input_length=10, mask_zero=True))(char_in)
        # character LSTM to get word encoding by characters
        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(emb_char)

        # main LSTM
        x = concatenate([inputs, char_enc])
        x = Bidirectional(LSTM(units=32, return_sequences=True,
                                     recurrent_dropout=0.1))(x)  # variational biLSTM
        # outputs = Bidirectional(LSTM(units=512, return_sequences=True,
        #                    recurrent_dropout=0.2, dropout=0.2))(outputs)
        outputs = TimeDistributed(Dense(self.n_labels, activation="softmax"))(x)  # a dense layer as suggested by neuralNer
        # crf = CRF(self.n_labels)  # CRF layer
        # outputs = crf(outputs)  # output

        model = Model(inputs=[inputs, char_in], outputs=outputs)
        model.compile(optimizer="adam", metrics=self.metrics,
                      loss=categorical_crossentropy)
        model.summary()
        self.model = model

    def preprocessing(self, features, labels):
        """
        Handles the preprocessing step. The features and labels are converted in vectors
            and their shape is adjusted.
        """
        X = self.preprocess_features(features)
        y = self.preprocess_labels(labels)
        # self.get_weights(labels)
        return X, y

    def get_sentences(self, collection: Collection):
        """
        Giving a collection, the features and labels of its sentences are returned
        """
        features = []
        labels = []
        X_char = []
        self.char2idx = get_char2idx(collection)
        for sentence in collection:
            feat, chars, label = get_instances(sentence, self.char2idx, labels=True)
            features.append(feat)
            labels.append(label)
            X_char.append(np.array(chars))
        return features, X_char, labels

    def get_features(self, collection: Collection):
        """Giving a collection, the features of its sentences are returned"""
        features = []
        X_char = []
        for sentence in collection:
            feat, chars = get_instances(sentence, self.char2idx, False)
            features.append(feat)
            X_char.append(chars)
        return features, X_char

    def fit_model(self, X, y, plot=False):
        """
        The model is fitted. The training begins
        """
        # hist = self.model.fit(X, y, batch_size=32, epochs=5,
        #             validation_split=0.2, verbose=1)
        # hist = self.model.fit(MyBatchGenerator(X, y, batch_size=30), epochs=5)
        X, X_char = X
        num_examples = len(X)
        steps_per_epoch = num_examples / 5
        # self.model.fit(self.generator(X, y), steps_per_epoch=steps_per_epoch, epochs=5)
        x_shapes, x_char_shapes, y_shapes = train_by_shape(X, y, X_char)
        for shape in x_shapes:
            self.model.fit(
                (np.asarray(x_shapes[shape]), np.asarray(x_char_shapes[shape])),
                np.asarray(y_shapes[shape]),
                epochs=5)

    def test_model(self, collection: Collection) -> Collection:
        collection = collection.clone()
        features, X_char,  = self.get_features(collection)
        X = self.preprocess_features(features, train=False)
        x_shapes, x_char_shapes, indices = predict_by_shape(X, X_char)
        pred = []
        for x_items, x_chars in zip(x_shapes, x_char_shapes):
            pred.extend(self.model.predict((np.asarray(x_items), np.asarray(x_chars))))
        labels = self.convert_to_label(pred)
        postprocessing_labels(labels, indices, collection)
        return collection

    def eval(self, path: Path, submit: Path):
        folder = 'scenario2-taskA'
        scenario = path / 'scenario2-taskA'
        print(f"Evaluating on {scenario}")

        input_data = Collection().load(scenario / "input.txt")
        print(f'Loaded {len(input_data)} input sentences')
        output_data = self.test_model(input_data)

        print(f"Writing output to {submit / folder}")
        output_data.dump(submit / folder / "output.txt", skip_empty_sentences=False)

    def save_model(self, name):
        BaseClassifier.save_model(self, name)
        json.dump(self.char2idx, open(fr'resources/{name}_charmap.json', 'w'))

    def load_model(self, name):
        BaseClassifier.load_model(self, name)
        self.char2idx = json.load(open(fr'resources/{name}_charmap.json', 'r'))


if __name__ == "__main__":
    collection = Collection().load_dir(Path('2021/ref/training'))
    # dev_set = Collection().load_dir(Path('2021/eval/develop/scenario1-main'))
    ner_clf = NERClassifier()
    ner_clf.train(collection)
    ner_clf.save_model('ner')
    # ner_clf.load_model('ner')
    ner_clf.eval(Path('2021/eval/develop/'), Path('2021/submissions/ner/develop/run1'))
    score.main(Path('2021/eval/develop'),
               Path('2021/submissions/ner/develop'),
               runs=[1], scenarios=[2], verbose=True, prefix="")
