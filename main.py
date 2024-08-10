from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

app = FastAPI()

class InputData(BaseModel):
    sentences: list[str]

# Load the Keras model
model = tf.keras.models.load_model('bangla_pos_ner_model.h5')

# Load tokenizer and max_len from a JSON file
with open('tokenizer.json') as f:
    tokenizer_data = json.load(f)

word_tokenizer = tokenizer_from_json(tokenizer_data['word_tokenizer'])
pos_tokenizer = tokenizer_from_json(tokenizer_data['pos_tokenizer'])
ner_tokenizer = tokenizer_from_json(tokenizer_data['ner_tokenizer'])
max_len = tokenizer_data['max_len']

def preprocess_input(sentences):
    sequences = word_tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

def infer(input_data):
    # Perform inference using the Keras model
    predictions = model.predict(input_data)
    return predictions

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bangla POS and NER Inference API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Preprocess input
        input_data_processed = preprocess_input(input_data.sentences)

        # Perform inference
        pos_pred, ner_pred = infer(input_data_processed)

        # Convert predictions from indices to tags
        pos_pred_labels = np.argmax(pos_pred, axis=-1)
        ner_pred_labels = np.argmax(ner_pred, axis=-1)

        pos_tags_list = []
        ner_tags_list = []

        for i in range(len(input_data.sentences)):
            pos_tags = [pos_tokenizer.index_word[tag] for tag in pos_pred_labels[i] if tag != 0]
            ner_tags = [ner_tokenizer.index_word[tag] for tag in ner_pred_labels[i] if tag != 0]
            pos_tags_list.append(pos_tags)
            ner_tags_list.append(ner_tags)

        return {
            "sentences": input_data.sentences,
            "pos_tags": pos_tags_list,
            "ner_tags": ner_tags_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
