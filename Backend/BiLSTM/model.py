import tensorflow as tf
from tensorflow.keras.layers import Input, Masking, Embedding, Bidirectional, LSTM, Dense, Attention, TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import numpy as np

from utils import train_val_split


class BiLSTMAttention:
    def __init__(self, X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers, dropout, recurrent_dropout):
        self.X_vocab_len = X_vocab_len
        self.X_max_len = X_max_len
        self.y_vocab_len = y_vocab_len
        self.y_max_len = y_max_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.model = None
        
    def create_model(self):
        # Define the input tensor
        inputs = Input(shape=(self.X_max_len,), dtype='int32')

        # Add masking layer to ignore unknown words
        masked_inputs = Masking(mask_value=1)(inputs)

        # Add embedding layer
        x = Embedding(input_dim=self.X_vocab_len, output_dim=self.hidden_size, mask_zero=True)(masked_inputs)

        # Add BiLSTM encoder layer
        for _ in range(self.num_layers):
            x = Bidirectional(LSTM(self.hidden_size, return_sequences=True, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout))(x)

        # Add attention layer
        query = x
        value = x
        attention = Attention()([query, value])

        # Add output layer
        outputs = TimeDistributed(Dense(self.y_vocab_len, activation='softmax'))(attention)

        # Create the model
        self.model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Print the model summary
        self.model.summary()
    
    def load(self, filepath):
        self.model = load_model(filepath)
        self.model.summary()
    
    def train_step(self, X_train, y_train, batch_size, epochs, validation_split, wandb_callback):
        y_train_one_hot = np.array(to_categorical(y_train, num_classes=self.model.output_shape[-1])) # one hot encoding

        X_train = np.array(X_train)

        X_train, y_train_one_hot, X_val, y_val_one_hot = train_val_split(X_train, y_train_one_hot, validation_split)

        self.model.fit(X_train, y_train_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val_one_hot), callbacks=[wandb_callback])

    def save(self, filepath):
        self.model.save(filepath)