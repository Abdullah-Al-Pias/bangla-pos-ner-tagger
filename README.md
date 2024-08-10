
# Bangla POS and NER Tagging System

This repository contains the code for a Bangla Parts of Speech (POS) and Named Entity Recognition (NER) tagging system. The system is built using TensorFlow/Keras for training the model and FastAPI for serving the model via an inference API.


## Table of Contents

- Project Overview
- Project Structure
- Prerequisites
- Installation
- Usage
- Data Preprocessing
- Model Training
- Model Evaluation
- Model Deployment
- ONNX Conversion
- Inference with ONNX model
- Running the API
- Testing the API
- Challenges
- Contributing
- License


## Project Overview
The objective of this project is to create an end-to-end system for performing POS and NER tagging on Bangla text. The project involves:

- Preprocessing the dataset to convert sentences and their corresponding tags into a format suitable for model training.
- Developing a Bidirectional LSTM model to perform POS and NER tagging.
- Deploying the trained model as a web service using FastAPI.
- Providing a script for converting the trained Keras model to ONNX format for potential integration into different platforms.

## Project Structure

Script for preprocessing the dataset
```bash
  data_preprocessing.py 
```

Script for training the Bidirectional LSTM model

```bash
  model_training.py 
```

Script for evaluating the model

```bash
  model_evalutation.py 
```

Script for deplyoing the model

```bash
  model_deployment.py 
```
Script for converting the Keras model to ONNX format

```bash
  convert_to_onnx.py 
```
Script for deplyoing the onnx model

```bash
  onnx_inference.py 
```
FastAPI server for serving the trained model
```bash
  main.py 
```
Python dependencies
```bash
  requirements.txt 
```
Dataset
```bash
  data.tsv
```
JSON file containing tokenizers and max_len
```bash
  tokenizer.json
```

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.8 or later
- TensorFlow 2.x
- FastAPI
- Uvicorn (for serving the FastAPI application)
- ONNX and ONNX Runtime (for ONNX conversion and inference)
## Installation

### Clone the repository:
```bash
git clone https://github.com/Abdullah-Al-Pias/bangla-pos-ner-tagger
```

### Install dependencies:
```bash
pip install -r requirements.txt
```
    
## Usage
### Data Preprocessing
To preprocess the data, use the data_preprocessing.py script:
```bash
python data_preprocessing.py
```
This script will generate the .npz file needed for training and a tokenizer.json file used during inference.

### Model Training
To train the model, you need to have a preprocessed dataset saved as a .npz file. The dataset should contain sequences of tokenized sentences along with their corresponding POS and NER tags. You can generate this file using the data_preprocessing.py script.

Once you have the dataset, run the model_training.py script to train the model:
```bash
python model_training.py
```
This script will load the preprocessed data, train the model, and save the trained model as 'bangla_pos_ner_model.h5'
### Model Evaluation
To evaluate the perfomance of the trained model run the model_evaluation.py script:
```bash
python model_evaluation.py
```
This will print out the accuracy, precision, recall, and F1 scores for both POS and NER tasks.

### Model Deployment
To run this script and get POS and NER tags for Bangla sentences, you can use the following command in your terminal:
```bash
python model_deploymnet.py --sentences "আপনার প্রথম বাক্য" "আপনার দ্বিতীয় বাক্য"
``` 
The script checks if any sentences were provided as arguments. If so, it prints the predicted POS and NER tags for each sentence.

### ONNX Conversion
If you want to convert the trained Keras model to ONNX format, use the onnx_conversion.py script:
```bash
python convert_to_onnx.py
```
This will create an ONNX model file 'bangla_pos_ner_model.onnx'

### Inference with ONNX model
To make inference with ONNX mode run the following script:
```bash
python onnx_inference.py
```
The script will output the POS and NER tags for each input sentence, as mapped from the ONNX model predictions.

### Running the API
To run the API, save the updated code in a file named main.py, and execute the following command in your terminal:
```bash
uvicorn main:app --reload
```
### Testing the API
- ### Health Check:

```bash
Check the health of the API by visiting http://localhost:8000/health.
```
- ### Making Predictions:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"sentences": ["আপনার প্রথম বাক্য", "আপনার দ্বিতীয় বাক্য"]}'
```

## Challenges
Some of the key challenges faced during the development of this project include:

- Handling imbalanced classes, especially in NER, where certain entity types are rare.
- Ensuring consistency between the training and inference processes, particularly with respect to tokenization and padding.
- Managing the transition from a TensorFlow/Keras model to ONNX format, which required careful handling of model inputs and outputs.

## Contributing
Contributions are welcome! If you would like to contribute, please open an issue or submit a pull request.


## License
This project is licensed under the MIT License. See the [License](https://choosealicense.com/licenses/mit/) file for details.

