import os
import librosa
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

from datasets import load_dataset

ds = load_dataset("CSALT/deepfake_detection_dataset_urdu")
audio_data = ds['train']
print(audio_data)

# Load the dataset and print a single example to see its structure
print("Sample example structure:")
print(audio_data[0])  # Inspect the structure of data

# Modified extract_mfcc function to work with the dataset structure
def extract_mfcc(audio_data):
    # The audio data in this dataset likely has array and sampling_rate fields
    y = audio_data['array']  # Get audio waveform
    sr = audio_data['sampling_rate']  # Get sample rate
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Create features and labels
features = []
# Since this is a deepfake detection dataset, we need to determine the labels
# You might need to derive labels from metadata or filename patterns
# For this example, let's assume we can determine labels from a pattern
# (You'll need to adjust this based on the actual dataset)

labels = []

for i, example in enumerate(audio_data):
    try:
        mfcc = extract_mfcc(example['audio'])
        features.append(mfcc)
        
        # You'll need to determine the correct way to get labels
        # This is just a placeholder - replace with actual label logic
        # For example:
        # - Check if 'filename' or 'path' contains 'deepfake' or 'bonafide'
        # - Use a predefined mapping based on file index
        # - Look for metadata in the dataset
        
        # Placeholder (assuming alternating labels for demonstration)
        label = 'bonafide' if i % 2 == 0 else 'deepfake'
        labels.append(label)
    except Exception as e:
        print(f"Error processing example {i}: {e}")

X = np.array(features)
y = np.array(labels)


#Encode Labels and Train-Test Split
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Bonafide = 0, Deepfake = 1
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.joblib")  # Save scaler



#Train SVM
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("SVM Report:")
print(classification_report(y_test, y_pred_svm))
print("ROC AUC:", roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1]))

joblib.dump(svm_model, "svm_model.joblib")


#Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1]))

joblib.dump(lr_model, "logistic_model.joblib")



#Train Single-Layer Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_pred_perc = perceptron.predict(X_test)

print("Perceptron Report:")
print(classification_report(y_test, y_pred_perc))
print("ROC AUC:", roc_auc_score(y_test, perceptron.predict_proba(X_test)[:, 1]))



#Train Deep Neural Network
dnn_model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
y_pred_dnn = dnn_model.predict(X_test).ravel()
y_pred_dnn_labels = (y_pred_dnn > 0.5).astype(int)

print("DNN Report:")
print(classification_report(y_test, y_pred_dnn_labels))
print("ROC AUC:", roc_auc_score(y_test, y_pred_dnn))

dnn_model.save("dnn_model.h5")



#Summary Table
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model_name, y_true, y_pred, y_proba=None):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba) if y_proba is not None else "N/A"
    }

results = [
    evaluate("SVM", y_test, y_pred_svm, svm_model.predict_proba(X_test)[:, 1]),
    evaluate("Logistic Regression", y_test, y_pred_lr, lr_model.predict_proba(X_test)[:, 1]),
    evaluate("Perceptron", y_test, y_pred_perc),
    evaluate("DNN", y_test, y_pred_dnn_labels, y_pred_dnn)
]

pd.DataFrame(results)
