from anntools import Collection

from pathlib import Path
# from typing import List

from base_clsf import BaseClassifier
from re_utils import get_instances, postprocessing_labels, predict_by_shape, train_by_shape
import score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Input, Embedding, Lambda
from tensorflow.keras.losses import categorical_crossentropy
from utils import weighted_loss
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
        self.get_model()
        return self.fit_model(X, y)

    def get_model(self):
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
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", metrics=self.metrics,
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
        self.get_weights(labels)
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

    def fit_model(self, X, y, plot=False):
        """
        The model is fitted. The training begins
        """
        # hist = self.model.fit(X, y, batch_size=32, epochs=5,
        #             validation_split=0.2, verbose=1)
        # hist = self.model.fit(MyBatchGenerator(X, y, batch_size=30), epochs=5)

        num_examples = len(X)
        steps_per_epoch = num_examples / 5
        # self.model.fit(self.generator(X, y), steps_per_epoch=steps_per_epoch, epochs=5)
        x_shapes, y_shapes = train_by_shape(X, y)
        for shape in x_shapes:
            self.model.fit(
                np.asarray(x_shapes[shape]),
                np.asarray(y_shapes[shape]),
                epochs=5)

    def test_model(self, collection: Collection) -> Collection:
        features = self.get_features(collection)
        X = self.preprocess_features(features, train=False)
        x_shapes, indices = predict_by_shape(X)
        pred = []
        for x_items in x_shapes:
            pred.extend(self.model.predict(np.asarray(x_items)))
        labels = self.convert_to_label(pred)
        postprocessing_labels(labels, indices, collection)
        return collection

    def eval(self, path: Path, submit: Path):
        folder = 'scenario3-taskB'
        scenario = path / folder
        print(f"Evaluating on {scenario}")

        input_data = Collection().load(scenario / "input.txt")
        print(f'Loaded {len(input_data)} input sentences')
        output_data = self.test_model(input_data)

        print(f"Writing output to {submit / folder}")
        output_data.dump(submit / folder / "output.txt", skip_empty_sentences=False)


if __name__ == "__main__":
    collection = Collection().load_dir(Path('2021/ref/training'))
    dev_set = Collection().load(Path('2021/eval/develop/scenario1-main/output.txt'))
    re_clf = REClassifier()
    re_clf.train(collection)
    re_clf.save_model('re')
    # ner_clf.load_model('re')
    re_clf.eval(Path('2021/eval/develop/'), Path('2021/submissions/ner/develop/run1'))
    score.main(Path('2021/eval/develop'),
               Path('2021/submissions/ner/develop'),
               runs=[1], scenarios=[3], verbose=True, prefix="")
