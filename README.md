# 🤖 ML Classifier Dashboard

![Project Banner](https://img.shields.io/badge/ML%20Classifier-Dashboard-1E88E5?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAABWUlEQVRIie2UvUoDQRDHf3O5NNECLMTCB7CwshJs7H0DKxtfQJ9BsLGxs/MNrK18ABsLwdZCJJVpInKXZCwWd0l2L4kWKf3D3rK7M/vb+ZgT/hpRN7BVTZrAGbB+L2ZAAlSdgpJwGjtbtA84Bh7m/mRu2QcwBT6AnEvQAmLxsQX0gGd5LoAtF2AdiIA3mY8EtAU8iPYMGMi6tKeEY2DqvPgKjIAzYFE0AwGMpG1aQLGkS1t1xgF4EWBfVhN4FO1VaV/uK4xMwIKA9g1QBVhR68YHsAjAFfqVJD4Al1jHDaDS1vJeS5y8voBT5RzHR9CS9Uv9RXeUvs8m5r0XsKR8WoUyOx94i1WFTVCT9k22PxGgp3TXB7gT47YPEAPXyulCCdgAWj6AfdMbMcbS7wB7YnznA9iY27ONmFtiTGNeiTZQwKWY3ZOdw3/4nfgEUghNrB8hJ5AAAAAASUVORK5CYII=)
[![PyPI - Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9%2B-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.19%2B-red)](https://streamlit.io/)

A comprehensive machine learning application with a dual-purpose focus: detecting Urdu deepfake audio and predicting software defect types. This project uses multiple ML algorithms (SVM, Logistic Regression, and Deep Neural Networks) with an interactive Streamlit interface for real-time predictions.

## ✨ Features

### 🎙️ Urdu Deepfake Audio Detection
- Audio file upload or real-time recording via microphone
- Feature extraction using MFCCs (Mel-Frequency Cepstral Coefficients)
- Real-time prediction with confidence scores
- Audio visualizations (waveform and spectrogram)
- Model selection between SVM, Logistic Regression, and DNN

### 📊 Software Defect Prediction
- Multi-label classification of software bug reports
- Predicts 7 defect types (blocker, regression, bug, documentation, enhancement, task, dependency upgrade)
- Visualized confidence scores for each category
- Interactive charts showing prediction probabilities


## 🛠️ Technical Architecture

### Data Processing
- **Audio Processing**: librosa for feature extraction (MFCCs, spectrograms)
- **Text Processing**: TF-IDF vectorization for bug report text analysis
- **Feature Scaling**: Standardization of numerical features

### Models
- **Support Vector Machine (SVM)**: For both classification tasks
- **Logistic Regression**: Binary and multi-label versions
- **Deep Neural Network (DNN)**: Custom architectures with multiple hidden layers

### Evaluation Metrics
- **Deepfake Detection**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Defect Prediction**: Hamming Loss, Micro-F1, Macro-F1, Precision@k

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: tensorflow, sklearn, librosa, pandas, numpy, streamlit, matplotlib, joblib
- Audio recording capability for the deepfake detection module

### Installation

```bash
# Clone the repository
git clone https://github.com/mnouman04/UrduFakeDetect-and-BugPredictor.git

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Using Pre-trained Models
The application uses pre-trained models by default. If you want to train your own models:

1. For Deepfake Detection:
   - Follow the notebook `notebooks/urdu_deepfake_detection.ipynb`
   - Use the Urdu Deepfake Detection Dataset: `CSALT/deepfake_detection_dataset_urdu`

2. For Software Defect Prediction:
   - Follow the notebook `notebooks/software_defect_prediction.ipynb`
   - Use the provided CSV data in `data/software_defects.csv`

## 📊 Model Performance

### Deepfake Detection

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| SVM   | 0.97     | 0.96      | 0.98   | 0.97     | 0.99    |
| LR    | 0.79     | 0.78      | 0.81   | 0.80     | 0.88    |
| DNN   | 0.98     | 0.98      | 0.98   | 0.98     | 0.99    |

### Software Defect Prediction

| Model | Hamming Loss | Micro-F1 | Macro-F1 | Precision@3 |
|-------|--------------|----------|----------|-------------|
| SVM   | 0.09         | 0.82     | 0.52     | 0.91        |
| LR    | 0.11         | 0.79     | 0.36     | 0.89        |
| DNN   | 0.11         | 0.79     | 0.47     | 0.94        |


## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

### 🌐 For Blog and LinkedIn post Visit

- 💼 [LinkedIn](https://www.linkedin.com/posts/mnouman4_detecting-deep-fake-voices-and-predicting-activity-7324802335040274432-CNUU?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAFWXAUYBII-iaDiJQT1gWv-KC1TIySc3Zd4)
- ✍️ [Medium](https://medium.com/@muhnouman88/detecting-deep-fake-voices-and-predicting-software-defects-a-dual-deep-learning-approach-5373ad62535a)

## 🙏 Acknowledgements
- [Hugging Face Datasets](https://huggingface.co/datasets) for the Urdu Deepfake Detection Dataset
- [audio_recorder_streamlit](https://github.com/stefanrmmr/audio_recorder_streamlit) for the audio recording component
- [Streamlit](https://streamlit.io/) for the amazing web framework