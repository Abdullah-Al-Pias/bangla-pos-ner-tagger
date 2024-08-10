import numpy as np
import onnxruntime as ort
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_model(onnx_model_path):
    # Load the ONNX model
    session = ort.InferenceSession(onnx_model_path)
    return session

def preprocess_input(sentences, word_tokenizer, max_len):
    # Convert sentences to sequences using the tokenizer
    sequences = word_tokenizer.texts_to_sequences(sentences)
    # Pad sequences to the maximum length
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    # Convert the padded sequences to float32
    padded_sequences = padded_sequences.astype('float32')
    return padded_sequences

def infer(session, input_data):
    # Prepare the input dictionary for ONNX runtime
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # Run inference
    outputs = session.run(output_names, {input_name: input_data})
    return outputs

def main(sentences, max_len, word_tokenizer):
    # Load the ONNX model
    session = load_model('bangla_pos_ner_model.onnx')

    # Preprocess input
    input_data = preprocess_input(sentences, word_tokenizer, max_len)

    # Perform inference
    pos_pred, ner_pred = infer(session, input_data)

    # Convert predictions from indices to tags
    pos_pred_labels = np.argmax(pos_pred, axis=-1)
    ner_pred_labels = np.argmax(ner_pred, axis=-1)

    pos_tags_list = []
    ner_tags_list = []

    for i in range(len(sentences)):
        pos_tags = pos_pred_labels[i].tolist()
        ner_tags = ner_pred_labels[i].tolist()
        pos_tags_list.append(pos_tags)
        ner_tags_list.append(ner_tags)

    return pos_tags_list, ner_tags_list

if __name__ == "__main__":
    import argparse
    from tensorflow.keras.preprocessing.text import Tokenizer

    # Load your tokenizer and max_len here (assuming they were saved during preprocessing)
    word_tokenizer = Tokenizer()  # Load your tokenizer here
    max_len = 50  # Set your max_len here

    parser = argparse.ArgumentParser(description='ONNX Inference for Bangla POS and NER Tagger')
    parser.add_argument('--sentences', type=str, nargs='+', help='Input sentence(s) for POS and NER tagging')
    args = parser.parse_args()

    if args.sentences:
        pos_tags, ner_tags = main(args.sentences, max_len, word_tokenizer)
        for i, sentence in enumerate(args.sentences):
            print(f"Sentence: {sentence}")
            print("POS Tags:", pos_tags[i])
            print("NER Tags:", ner_tags[i])
