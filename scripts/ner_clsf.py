from anntools import Collection
from pathlib import Path

from base_clsf import BaseClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Input, Embedding, Lambda
from tensorflow.keras.losses import categorical_crossentropy
from ner_utils import get_instances, postprocessing_labels
# from utils import DataGeneratorPredict
# import tensorflow_addons as tfa
from tensorflow_addons.layers import CRF

# from keras_crf import CRF
import numpy as np
from utils import predict_by_shape, weighted_loss



class NERClassifier(BaseClassifier):
    """Classifier for the name entity resolution task"""

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
        # outputs = TimeDistributed(Dense(self.n_labels, activation="softmax"))(
        #     outputs)  # a dense layer as suggested by neuralNer
        # crf = CRF(self.n_labels)  # CRF layer
        # outputs = crf(outputs)  # output

        model = Model(inputs=inputs, outputs=outputs)
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
        for sentence in collection:
            feat, label = get_instances(sentence, labels=True)
            features.append(feat)
            labels.append(label)
        return features, labels

    def get_features(self, collection: Collection):
        """Giving a collection, the features of its sentences are returned"""
        return [get_instances(sentence, False) for sentence in collection]

    def test_model(self, collection: Collection) -> Collection:
        collection = collection.clone()
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
        folder = 'scenario2-taskA'
        scenario = path / 'scenario2-taskA'
        print(f"Evaluating on {scenario}")

        input_data = Collection().load(scenario / "input.txt")
        print(f'Loaded {len(input_data)} input sentences')
        output_data = self.test_model(input_data)

        print(f"Writing output to {submit / folder}")
        output_data.dump(submit / folder / "output.txt", skip_empty_sentences=False)


if __name__ == "__main__":
    collection = Collection().load_dir(Path('2021/ref/training'))
    # dev_set = Collection().load_dir(Path('2021/eval/develop/scenario1-main'))
    ner_clf = NERClassifier()
    ner_clf.train(collection)
    ner_clf.save_model('ner')
    # ner_clf.load_model('ner')
    ner_clf.eval(Path('2021/eval/develop/'), Path('2021/submissions/ner/develop/run1'))
