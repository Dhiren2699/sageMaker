from flask import Flask, jsonify
import os
import boto3
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mne.io import read_raw_edf, read_raw_bdf

app = Flask(__name__)

# S3 Configuration
bucket_name = 'dhiren-dorich-domain-406'
annotations_key = 'annotations_2017_A_fixed.csv'
edf_key = 'eeg13.edf'
model_key = 'my_aws.keras'

# Local Paths
edf_path = '/tmp/eeg13.edf'
model_path = '/tmp/model.keras'
annotations_path = '/tmp/annotations.csv'

# Initialize S3 Client
s3 = boto3.client('s3')

# Download and Load Model and Annotations at Startup
if not os.path.exists(model_path):
    s3.download_file(bucket_name, model_key, model_path)
model = load_model(model_path)

if not os.path.exists(annotations_path):
    s3.download_file(bucket_name, annotations_key, annotations_path)
annA = pd.read_csv(annotations_path)

def summary_eeg_data(file_path):
    if file_path.lower().endswith('.edf'):
        raw = read_raw_edf(file_path, preload=True, verbose=False)
    elif file_path.lower().endswith('.bdf'):
        raw = read_raw_bdf(file_path, preload=True, verbose=False)
    else:
        raise ValueError("Unsupported file format. Only .edf and .bdf files are supported.")

    data = raw.get_data()
    eeg_file_number = file_path.split('/')[-1].split('.')[0].split('g')[1]
    annotations = annA[eeg_file_number].dropna().values

    window_size = 256
    num_samples = data.shape[1]
    windows = []

    for i in range(len(annotations)):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        if end_idx <= num_samples:
            windows.append(data[:, start_idx:end_idx])
        else:
            break

    return np.array(windows), np.array(annotations).reshape(-1, 1)

@app.route('/predict', methods=['GET'])
def predict_eeg():
    # Download EEG file
    s3.download_file(bucket_name, edf_key, edf_path)

    # Process EEG Data
    window_data, labels = summary_eeg_data(edf_path)

    # Make Predictions
    Y_PRED = model.predict(window_data)
    Y_PRED = (Y_PRED > 0.5).astype("int32")

    # Calculate Metrics
    accuracy = accuracy_score(labels, Y_PRED)
    precision = precision_score(labels, Y_PRED)
    recall = recall_score(labels, Y_PRED)
    f1 = f1_score(labels, Y_PRED)

    # Return Results
    results = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
