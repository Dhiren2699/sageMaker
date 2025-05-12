import os
import boto3
import numpy as np
import pandas as pd
import warnings
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mne.io import read_raw_edf, read_raw_bdf

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    print("Downloading model from S3...")
    s3.download_file(bucket_name, model_key, model_path)
model = load_model(model_path)
print("Model loaded.")

if not os.path.exists(annotations_path):
    print("Downloading annotations from S3...")
    s3.download_file(bucket_name, annotations_key, annotations_path)
annA = pd.read_csv(annotations_path)
print("Annotations loaded.")

def summary_eeg_data(file_path):
    print(f"Reading EEG file: {file_path}")
    if file_path.lower().endswith('.edf'):
        raw = read_raw_edf(file_path, preload=True, verbose=False)
    elif file_path.lower().endswith('.bdf'):
        raw = read_raw_bdf(file_path, preload=True, verbose=False)
    else:
        raise ValueError("Unsupported file format. Only .edf and .bdf files are supported.")

    data = raw.get_data()
    print("EEG data shape:", data.shape)

    eeg_file_number = file_path.split('/')[-1].split('.')[0].split('g')[1]
    annotations = annA[eeg_file_number].dropna().values
    print("Number of annotations:", len(annotations))

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

    print("Total EEG windows created:", len(windows))
    return np.array(windows), np.array(annotations).reshape(-1, 1)

def predict_eeg():
    print("Downloading EEG file from S3...")
    s3.download_file(bucket_name, edf_key, edf_path)

    print("Processing EEG data...")
    window_data, labels = summary_eeg_data(edf_path)
    print("Window data shape:", window_data.shape)
    print("Labels shape:", labels.shape)

    # Predict timing
    pred_start_time = time.time()
    print("Making predictions...")
    Y_PRED = model.predict(window_data)
    pred_end_time = time.time()

    print("Raw prediction output (first 5):", Y_PRED[:5].flatten())
    Y_PRED = (Y_PRED > 0.5).astype("int32")
    print("Binarized predictions:", np.unique(Y_PRED, return_counts=True))

    # Calculate Metrics
    accuracy = accuracy_score(labels, Y_PRED)
    precision = precision_score(labels, Y_PRED, zero_division=0)
    recall = recall_score(labels, Y_PRED, zero_division=0)
    f1 = f1_score(labels, Y_PRED, zero_division=0)

    # Show timing
    pred_duration = pred_end_time - pred_start_time
    print(f"Time taken for prediction: {pred_duration:.2f} seconds")

    # Return Results
    results = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "prediction_time_sec": round(pred_duration, 2)
    }

    print("Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v}")

    #return results
    return results

if __name__ == '__main__':
    start_time = time.time()
    predict_eeg()
    total_time = time.time() - start_time
    print(f"\nTotal script execution time: {total_time:.2f} seconds")
