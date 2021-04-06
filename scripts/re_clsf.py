from anntools import Collection, Keyphrase, Relation

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Input, Embedding, Lambda
import tensorflow_addons as tfa
from tensorflow_addons.layers import CRF

class REClassifier:
    "Classifier for the relation extraction task"
    def __init__(self):
        pass
   
    def fit(self, collection: Collection):
        keyphrases = self._get_keyphrases(collection)
        # train with the gold keyphrases

    def get_bi_lstm_model(self, n_timesteps:int, mode:str):
        inputs = Input(shape=(None,), dtype='int32')
        outputs = Embedding(100, 40, trainable=True, mask_zero=True)(inputs)
        outputs = Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=mode)(outputs)
        outputs = TimeDistributed(Dense(1, activation='sigmoid'))(outputs)
        
        sequence_mask = Lambda(lambda x: tf.greater(x, 0))(inputs)

        crf = CRF(7)
        # outputs = CRF(7)(outputs)
        outputs = crf(outputs, mask=sequence_mask)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss='binary_crossentropy',
            metrics=['accuracy'],
            optimizer=tf.keras.optimizers.Adam(5e-5)
            )
        model.summary()


        # model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train_model(self, model, n_timesteps:int):
        loss = list()
        for _ in range(250):
            # generate new random sequence
            X,y = self.get_sentence(n_timesteps)
            # fit model for one epoch on this sequence
            hist = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
            loss.append(hist.history['loss'][0])
        return loss

    def get_sentence(self, n_timesteps):
        pass

    def _get_keyphrases(self, collection):
        "Gets gold keyphrases"
        keyphrases = {}
        for sentence in collection.sentences:
            for keyphrase in sentence.keyphrases:
                text = keyphrase.text.lower()
                keyphrases[text] = keyphrase.label
        return keyphrases

    def run(self, collection: Collection):
        collection = collection.clone()
        # returns a collection with everything annotated
        return collection


if __name__ == "__main__":
    clsf = REClassifier()
    model = clsf.get_bi_lstm_model(10, 'concat')
