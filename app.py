import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import tempfile
import os
import time
# Import the correct audio recorder component
from audio_recorder_streamlit import audio_recorder
# Set page configuration
st.set_page_config(
    page_title="Urdu Deepfake Audio Detection",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Apply custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5E35B1;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 0.5rem;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .warning-text {
        color: #FFC107;
        font-weight: bold;
    }
    .danger-text {
        color: #F44336;
        font-weight: bold;
    }
    .audio-input-section {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .recording-section {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)
# Function to load models
@st.cache_resource
def load_models():
    try:
        # Load models for deepfake detection
        svm_model_df = joblib.load("models/svm_model1.joblib")
        lr_model_df = joblib.load("models/logistic_model1.joblib")
        dnn_model_df = tf.keras.models.load_model("models/dnn_model1.h5")
        scaler = joblib.load("models/scaler1.joblib")
        
        # Load models for software defect prediction
        svm_model_sd = joblib.load("models/svm_model.pkl")
        lr_model_sd = joblib.load("models/logistic_model.pkl")
        mlp_model_sd = joblib.load("models/mlp_model.pkl")
        tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        
        return {
            "deepfake": {
                "SVM": svm_model_df,
                "Logistic Regression": lr_model_df,
                "DNN": dnn_model_df,
                "scaler": scaler
            },
            "defect": {
                "SVM": svm_model_sd,
                "Logistic Regression": lr_model_sd,
                "DNN": mlp_model_sd,
                "vectorizer": tfidf_vectorizer
            }
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None
# Function to extract features from audio
def extract_features(audio_data, sr):
    try:
        # Extract MFCCs - use exactly the same parameters as in training
        # This matches the extract_mfcc function used during model training
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        
        # Return just the MFCC features to match what the model expects
        return mfcc_mean.reshape(1, -1)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None
# Function to plot audio waveform
def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
# Function to plot spectrogram
def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel-frequency spectrogram')
    fig.tight_layout()
    return fig
# Main function
def main():
    # Load models
    models = load_models()
    if not models:
        st.error("Failed to load models. Please check that all model files are present.")
        return
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 Machine Learning Classifier Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Detect deepfake Urdu audio using machine learning</p>", unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎙️ Deepfake Audio Detection", "📝 Software Defect Prediction"])
    
    # Sidebar for common controls
    st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.sidebar.title("Settings")
    
    # Deepfake Audio Detection Tab
    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>🎙️ Urdu Deepfake Audio Detection</h2>", unsafe_allow_html=True)
        st.markdown("Determine if an Urdu audio sample is genuine (bonafide) or artificially generated (deepfake).")
        
        # Model selection in the tab
        df_model_option = st.selectbox(
            "Select Model for Deepfake Detection",
            ["Logistic Regression", "SVM", "DNN"],
            key="df_model"
        )
        
        # Audio input method selection
        input_method = st.radio(
            "Choose audio input method:",
            ["Upload Audio File", "Record Audio"],
            horizontal=True
        )
        
        audio_data = None
        
        if input_method == "Upload Audio File":
            st.markdown("<div class='audio-input-section'>", unsafe_allow_html=True)
            st.subheader("Upload Audio File")
            # Support multiple audio formats
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=["wav", "mp3", "ogg", "flac", "m4a"],
                key="audio_file"
            )
            
            if uploaded_file:
                st.audio(uploaded_file)
                
                # Save to temp file for processing
                file_extension = uploaded_file.name.split(".")[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                    
                # Load and process the audio
                try:
                    y, sr = librosa.load(audio_path, sr=16000)
                    audio_data = (y, sr, audio_path)
                    st.success(f"Audio file loaded successfully: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading audio file: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
                
        else:  # Record Audio
            st.markdown("<div class='recording-section'>", unsafe_allow_html=True)
            st.subheader("Record Audio")
            st.markdown("📢 Record your voice in Urdu for analysis")
            
            # Audio recorder component (using the correct import)
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                key="audio_recorder"
            )
            
            if audio_bytes:
                st.success("Audio recorded successfully!")
                st.audio(audio_bytes, format="audio/wav")
                
                # Save recorded audio to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                    audio_path = tmp_file.name
                    
                # Load and process the audio
                try:
                    y, sr = librosa.load(audio_path, sr=16000)
                    audio_data = (y, sr, audio_path)
                except Exception as e:
                    st.error(f"Error processing recorded audio: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Process audio data if available
        if audio_data:
            y, sr, audio_path = audio_data
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Extract features for prediction
                features = extract_features(y, sr)
                
                if features is not None:
                    # Scale features
                    scaled_features = models['deepfake']['scaler'].transform(features)
                    
                    # Select model based on user choice
                    model = models['deepfake'][df_model_option]
                    
                    # Make prediction - FIX: Handle different output formats from different models
                    if df_model_option == "DNN":
                        prediction = model.predict(scaled_features)[0]
                        
                        # Check the shape of the prediction to ensure proper handling
                        if isinstance(prediction, np.ndarray) and prediction.size > 1:
                            # For multi-class output - FIXED: Swapped the indices to correct the classification
                            # Assuming index 0 is for deepfake and index 1 is for bonafide in the DNN model's output
                            deepfake_conf = float(prediction[0])
                            bonafide_conf = float(prediction[1]) if len(prediction) > 1 else float(1 - prediction[0])
                        else:
                            # For single value output (binary classification)
                            # Since we've identified the DNN model is reversed, we swap the interpretation
                            deepfake_conf = float(prediction)
                            bonafide_conf = float(1 - prediction)
                    else:
                        # For traditional ML models (SVM, Logistic Regression)
                        try:
                            proba = model.predict_proba(scaled_features)[0]
                            # Check if we have two probabilities (binary classification)
                            if len(proba) >= 2:
                                bonafide_conf = float(proba[0])
                                deepfake_conf = float(proba[1])
                            else:
                                # If only one probability is returned, assume it's for the positive class
                                bonafide_conf = float(proba[0])
                                deepfake_conf = float(1 - proba[0])
                        except:
                            # Fallback if predict_proba is not available
                            prediction = model.predict(scaled_features)[0]
                            # Convert binary prediction to confidence (0 or 1)
                            bonafide_conf = 1.0 if prediction == 0 else 0.0
                            deepfake_conf = 1.0 if prediction == 1 else 0.0
                    
                    # Ensure confidence values are simple floats for string formatting
                    bonafide_conf = float(bonafide_conf)
                    deepfake_conf = float(deepfake_conf)
                    
                    # Determine the label based on confidence scores
                    label = "Bonafide" if bonafide_conf > deepfake_conf else "Deepfake"
                    
                    # Display visualizations
                    st.markdown("<h3>Audio Visualization</h3>", unsafe_allow_html=True)
                    
                    # Waveform
                    st.pyplot(plot_waveform(y, sr))
                    
                    # Spectrogram
                    st.pyplot(plot_spectrogram(y, sr))
                
            with col2:
                if features is not None:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
                    
                    # Display prediction with appropriate styling
                    if label == "Bonafide":
                        st.markdown(f"<h2 class='success-text'>✅ {label}</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h2 class='danger-text'>⚠️ {label}</h2>", unsafe_allow_html=True)
                    
                    # Confidence values
                    st.markdown("<h4>Confidence Scores</h4>", unsafe_allow_html=True)
                    
                    # Create progress bars for confidence
                    bonafide_conf_pct = bonafide_conf * 100
                    deepfake_conf_pct = deepfake_conf * 100
                    
                    st.markdown(f"**Bonafide**: {bonafide_conf_pct:.2f}%")
                    st.progress(bonafide_conf)
                    
                    st.markdown(f"**Deepfake**: {deepfake_conf_pct:.2f}%")
                    st.progress(deepfake_conf)
                    
                    # Display details about the model used
                    st.markdown("---")
                    st.markdown(f"**Model Used**: {df_model_option}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Clean up temp file
                    try:
                        os.unlink(audio_path)
                    except:
                        pass
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Software Defect Prediction Tab
    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>📝 Software Defect Prediction</h2>", unsafe_allow_html=True)
        st.markdown("Enter a bug report description to predict its type(s).")
        
        # Model selection in the tab
        sd_model_option = st.selectbox(
            "Select Model for Defect Prediction",
            ["Logistic Regression", "SVM", "DNN"],
            key="sd_model"
        )
        
        # Text input area
        input_text = st.text_area(
            "Bug Report Description", 
            height=150,
            placeholder="Enter a description of the software bug or issue here...",
            key="bug_report"
        )
        
        # Labels for software defect types
        labels = [
            'type_blocker', 'type_regression', 'type_bug', 'type_documentation',
            'type_enhancement', 'type_task', 'type_dependency_upgrade'
        ]
        
        # Button to trigger prediction
        if st.button("Predict Defect Types", key="predict_defects"):
            if not input_text.strip():
                st.warning("Please enter a report description.")
            else:
                # Transform text using TF-IDF
                X_vec = models['defect']['vectorizer'].transform([input_text])
                
                # Select model based on user choice
                model = models['defect'][sd_model_option]
                
                # Make prediction
                if sd_model_option == "DNN":
                    preds = model.predict_proba(X_vec.toarray())
                else:
                    preds = model.predict_proba(X_vec)
                
                preds = np.array(preds)
                preds_binary = (preds > 0.5).astype(int)
                
                # Organize results
                results = []
                for idx, label in enumerate(labels):
                    clean_label = label.replace('type_', '').capitalize()
                    confidence = preds[0][idx]
                    is_predicted = preds_binary[0][idx] == 1
                    results.append({
                        "label": clean_label,
                        "confidence": confidence,
                        "predicted": is_predicted
                    })
                
                # Sort by prediction status and then confidence
                results.sort(key=lambda x: (-int(x["predicted"]), -x["confidence"]))
                
                # Display results
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("<h3>Predicted Types</h3>", unsafe_allow_html=True)
                    
                    # Check if any type was predicted
                    if any(r["predicted"] for r in results):
                        for r in results:
                            if r["predicted"]:
                                st.markdown(
                                    f"<div class='metric-card'><h4 class='success-text'>✅ {r['label']}</h4>"
                                    f"Confidence: {r['confidence']*100:.2f}%</div>",
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("No specific defect types predicted with confidence > 50%")
                
                with col2:
                    st.markdown("<h3>Confidence Chart</h3>", unsafe_allow_html=True)
                    
                    # Create a bar chart
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    # Extract labels and confidence values
                    chart_labels = [r["label"] for r in results]
                    confidences = [r["confidence"] * 100 for r in results]
                    colors = ['#4CAF50' if r["predicted"] else '#F44336' for r in results]
                    
                    # Create horizontal bar chart
                    y_pos = np.arange(len(chart_labels))
                    ax.barh(y_pos, confidences, align='center', color=colors)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(chart_labels)
                    ax.invert_yaxis()  # labels read top-to-bottom
                    ax.set_xlabel('Confidence (%)')
                    ax.set_title('Defect Type Prediction Confidence')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Add a vertical line at 50% threshold
                    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
                    ax.text(51, len(chart_labels)-0.5, '50% threshold', 
                            verticalalignment='center', alpha=0.7)
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    st.markdown(f"**Model Used**: {sd_model_option}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application demonstrates machine learning models for:\n\n"
        "1. **Urdu Deepfake Audio Detection**: Distinguishes between genuine (bonafide) and artificially generated (deepfake) Urdu audio samples\n\n"
        "2. **Software Defect Prediction**: Classifies software bug reports into multiple categories based on text content"
    )
    
    st.sidebar.markdown("### Models Available")
    st.sidebar.markdown("""
    - **SVM**: Support Vector Machine
    - **Logistic Regression**: Linear model for classification
    - **DNN**: Deep Neural Network
    """)
if __name__ == "__main__":
    main()