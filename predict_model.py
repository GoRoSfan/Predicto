import csv
import random

from keras.layers import Embedding
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras


class PredictoModel:
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 100, max_len: int = 20,
                 model_path: str = 'model'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        self.model_path = model_path
        try:
            self._model = keras.models.load_model(model_path)
        except OSError:
            self._model = None

    @property
    def model(self):
        if self._model is None:
            raise ValueError("No model available.")
        return self._model

    @model.setter
    def model(self, value):
        if not isinstance(value, keras.Sequential):
            raise ValueError("Incorrect model type was set up. Please use 'keras.Sequential'.")
        self._model = value
        self._model.save("model")

    # @staticmethod
    # def _preprocess_claps(claps_raw: list):
    #     y = np.array(claps_raw)[:, np.newaxis]
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     return scaler.fit_transform(y)

    @staticmethod
    def get_text_vector(titles):
        max_features = 10000  # Maximum vocab size.
        max_len = 20  # Sequence length to pad the outputs to.

        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=max_features,
            output_mode='int',
            output_sequence_length=max_len)

        vectorize_layer.adapt(titles)
        return vectorize_layer

    def new_model(self, vocab_titles):
        model = keras.Sequential()
        model.add(self.get_text_vector(vocab_titles))
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_len))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(optimizer="adam",
                      loss="mse",
                      metrics=['mae', 'accuracy'])

        model.summary()

        self.model = model

    def update_model(self, titles: list, claps: list):
        X = np.array(titles)
        y = np.array([float(i) for i in claps])
        self.model.fit(X, y, epochs=50, batch_size=512, validation_split=0.2)

    def predict_claps(self, title: str):
        seed = sum(ord(char) for char in title)
        random.seed(seed)
        self.model.predict([title])
        return random.random()*1000

if __name__ == '__main__':
    train = list(csv.DictReader(open("train_set.csv", encoding="utf-8")))
    titles_ = [t["title"] for t in train]
    claps_ = [t["claps"] for t in train]
    train = list(csv.DictReader(open("medium_data.csv", encoding="utf-8")))
    titles_2 = [t["title"] for t in train]
    claps_2 = [t["claps"] for t in train]
    titles_.extend(titles_2)
    claps_.extend(claps_2)
    pm = PredictoModel()
    pm.new_model(titles_)
    pm.update_model(titles_, claps_)
    # print(pm.predict_claps("Look deeper"))