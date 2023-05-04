import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout


class PredictoModel:
    val_data = {
        "titles": ["Lost"],
        "views": [1200]
    }

    def __init__(self, vocab_size=5000, embedding_dim=100, max_len=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.model = None

    def new_model(self):
        # Define the model
        model = Sequential()

        # Add an embedding layer with input_dim as the vocabulary size and output_dim as the embedding dimension
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_len))

        # Add LSTM layer with 32 hidden units
        model.add(LSTM(units=32))

        # Add dropout layer to prevent overfitting
        model.add(Dropout(0.2))

        # Add dense output layer with one unit and sigmoid activation function
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model with binary cross-entropy loss and Adam optimizer
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def update_model(self, titles, views):
        # Train the model
        self.model.fit(titles, views, epochs=10, batch_size=32,
                       validation_data=(self.val_data["titles"], self.val_data["views"]))

if __name__ == '__main__':
    pm = PredictoModel()
    pm.new_model()
    x = np.expand_dims(["Dark themes", "Chalky"], axis=1)
    y = np.expand_dims([1000, 2000], axis=1)
    pm.update_model(x, y)