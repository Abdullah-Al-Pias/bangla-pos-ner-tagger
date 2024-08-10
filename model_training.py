import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Bidirectional
import tensorflow as tf

# Load preprocessed data
data = np.load('preprocessed_data.npz', allow_pickle=True)
X = data['X']
Y_pos = data['Y_pos']
Y_ner = data['Y_ner']
max_len = data['max_len'].item()
word_tokenizer = data['word_tokenizer'].item()
pos_tokenizer = data['pos_tokenizer'].item()
ner_tokenizer = data['ner_tokenizer'].item()

# Split the dataset
X_train, X_test, Y_train_pos, Y_test_pos, Y_train_ner, Y_test_ner = train_test_split(
    X, Y_pos, Y_ner, test_size=0.2, random_state=42
)
X_train, X_val, Y_train_pos, Y_val_pos, Y_train_ner, Y_val_ner = train_test_split(
    X_train, Y_train_pos, Y_train_ner, test_size=0.1, random_state=42
)

# Model architecture
input = Input(shape=(max_len,))
model = Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=128, input_length=max_len)(input)
model = Bidirectional(LSTM(units=256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(model)

# POS tagging output
pos_output = TimeDistributed(Dense(Y_train_pos.shape[2], activation='softmax'))(model)

# NER tagging output
ner_output = TimeDistributed(Dense(Y_train_ner.shape[2], activation='softmax'))(model)

model = Model(inputs=input, outputs=[pos_output, ner_output])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, [Y_train_pos, Y_train_ner],
    validation_data=(X_val, [Y_val_pos, Y_val_ner]),
    epochs=100,
    batch_size=64,
    verbose=1
)

# Save the model
model.save('bangla_pos_ner_model.h5')
