import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(file_path):
    sentences = []
    pos_tags = []
    ner_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    sentence = None
    tokens = []
    pos = []
    ner = []

    for line in lines:
        line = line.strip()
        if not line:
            if sentence is not None:
                sentences.append(sentence)
                pos_tags.append(pos)
                ner_tags.append(ner)
            sentence = None
            tokens = []
            pos = []
            ner = []
        elif sentence is None:
            sentence = line
        else:
            token, pos_tag, ner_tag = line.split('\t')
            tokens.append(token)
            pos.append(pos_tag)
            ner.append(ner_tag)

    if sentence is not None:
        sentences.append(sentence)
        pos_tags.append(pos)
        ner_tags.append(ner)

    return sentences, pos_tags, ner_tags

def preprocess_data(sentences, pos_tags, ner_tags):
    word_tokenizer = Tokenizer(lower=False)
    word_tokenizer.fit_on_texts(sentences)
    word_sequences = word_tokenizer.texts_to_sequences(sentences)

    pos_tokenizer = Tokenizer(lower=False)
    pos_tokenizer.fit_on_texts(pos_tags)
    pos_sequences = pos_tokenizer.texts_to_sequences(pos_tags)

    ner_tokenizer = Tokenizer(lower=False)
    ner_tokenizer.fit_on_texts(ner_tags)
    ner_sequences = ner_tokenizer.texts_to_sequences(ner_tags)

    max_len = max(len(seq) for seq in word_sequences)
    X = pad_sequences(word_sequences, maxlen=max_len, padding='post')
    Y_pos = pad_sequences(pos_sequences, maxlen=max_len, padding='post')
    Y_ner = pad_sequences(ner_sequences, maxlen=max_len, padding='post')

    num_pos_tags = len(pos_tokenizer.word_index) + 1
    num_ner_tags = len(ner_tokenizer.word_index) + 1

    Y_pos = to_categorical(Y_pos, num_classes=num_pos_tags)
    Y_ner = to_categorical(Y_ner, num_classes=num_ner_tags)

    return X, Y_pos, Y_ner, max_len, word_tokenizer, pos_tokenizer, ner_tokenizer

def save_tokenizers(word_tokenizer, pos_tokenizer, ner_tokenizer, max_len, output_file):
    tokenizer_data = {
        'word_tokenizer': word_tokenizer.to_json(),
        'pos_tokenizer': pos_tokenizer.to_json(),
        'ner_tokenizer': ner_tokenizer.to_json(),
        'max_len': max_len
    }
    with open(output_file, 'w') as f:
        json.dump(tokenizer_data, f)

if __name__ == "__main__":
    file_path = 'data.tsv'
    sentences, pos_tags, ner_tags = load_and_preprocess_data(file_path)
    X, Y_pos, Y_ner, max_len, word_tokenizer, pos_tokenizer, ner_tokenizer = preprocess_data(sentences, pos_tags, ner_tags)

    # Save the preprocessed data
    np.savez('preprocessed_data.npz', X=X, Y_pos=Y_pos, Y_ner=Y_ner, max_len=max_len,
             word_tokenizer=word_tokenizer, pos_tokenizer=pos_tokenizer, ner_tokenizer=ner_tokenizer)

    # Save the tokenizers and max_len as JSON
    save_tokenizers(word_tokenizer, pos_tokenizer, ner_tokenizer, max_len, 'tokenizer.json')
  
