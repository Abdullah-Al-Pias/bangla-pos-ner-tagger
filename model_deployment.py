import numpy as np
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_predict(sentences):
    data = np.load('preprocessed_data.npz', allow_pickle=True)
    max_len = data['max_len'].item()
    word_tokenizer = data['word_tokenizer'].item()
    pos_tokenizer = data['pos_tokenizer'].item()
    ner_tokenizer = data['ner_tokenizer'].item()
    
    model = load_model('bangla_pos_ner_model.h5')
    
    sequences = word_tokenizer.texts_to_sequences(sentences)
    X_new = pad_sequences(sequences, maxlen=max_len, padding='post')
    pos_pred, ner_pred = model.predict(X_new)
    
    pos_tags_list = []
    ner_tags_list = []
    
    for i in range(len(sentences)):
        pos_tags = np.argmax(pos_pred[i], axis=-1)
        ner_tags = np.argmax(ner_pred[i], axis=-1)
        pos_tags = [pos_tokenizer.index_word[tag] for tag in pos_tags if tag != 0]
        ner_tags = [ner_tokenizer.index_word[tag] for tag in ner_tags if tag != 0]
        pos_tags_list.append(pos_tags)
        ner_tags_list.append(ner_tags)
    
    return pos_tags_list, ner_tags_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bangla POS and NER Tagger')
    parser.add_argument('--sentences', type=str, nargs='+', help='Input sentence(s) for POS and NER tagging')
    args = parser.parse_args()

    if args.sentences:
        predicted_pos_tags, predicted_ner_tags = load_and_predict(args.sentences)
        for i, sentence in enumerate(args.sentences):
            print(f"Sentence: {sentence}")
            print("POS Tags:", predicted_pos_tags[i])
            print("NER Tags:", predicted_ner_tags[i])
