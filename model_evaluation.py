import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

# Load preprocessed data
data = np.load('preprocessed_data.npz', allow_pickle=True)
X = data['X']
Y_pos = data['Y_pos']
Y_ner = data['Y_ner']
max_len = data['max_len'].item()

# Load the model
model = load_model('bangla_pos_ner_model.h5')

# Predict on the test set
pos_pred, ner_pred = model.predict(X)

# Convert predictions from categorical to label indices
pos_pred_labels = np.argmax(pos_pred, axis=-1)
ner_pred_labels = np.argmax(ner_pred, axis=-1)
pos_true_labels = np.argmax(Y_pos, axis=-1)
ner_true_labels = np.argmax(Y_ner, axis=-1)

# Flatten the lists for metric calculations
pos_pred_flat = pos_pred_labels.flatten()
ner_pred_flat = ner_pred_labels.flatten()
pos_true_flat = pos_true_labels.flatten()
ner_true_flat = ner_true_labels.flatten()

# Filter out padding indices for evaluation
mask_pos = pos_true_flat != 0
mask_ner = ner_true_flat != 0
pos_pred_flat = pos_pred_flat[mask_pos]
pos_true_flat = pos_true_flat[mask_pos]
ner_pred_flat = ner_pred_flat[mask_ner]
ner_true_flat = ner_true_flat[mask_ner]

# Calculate metrics for POS tagging
pos_accuracy = accuracy_score(pos_true_flat, pos_pred_flat)
pos_precision = precision_score(pos_true_flat, pos_pred_flat, average='macro')
pos_recall = recall_score(pos_true_flat, pos_pred_flat, average='macro')
pos_f1 = f1_score(pos_true_flat, pos_pred_flat, average='macro')

# Calculate metrics for NER tagging
ner_accuracy = accuracy_score(ner_true_flat, ner_pred_flat)
ner_precision = precision_score(ner_true_flat, ner_pred_flat, average='macro')
ner_recall = recall_score(ner_true_flat, ner_pred_flat, average='macro')
ner_f1 = f1_score(ner_true_flat, ner_pred_flat, average='macro')

# Print results
print(f'POS Accuracy: {pos_accuracy}')
print(f'POS Precision: {pos_precision}')
print(f'POS Recall: {pos_recall}')
print(f'POS F1 Score: {pos_f1}')

print(f'NER Accuracy: {ner_accuracy}')
print(f'NER Precision: {ner_precision}')
print(f'NER Recall: {ner_recall}')
print(f'NER F1 Score: {ner_f1}')
