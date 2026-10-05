# WiFi CSI Human Activity Recognition

A device-free Human Activity Recognition (HAR) system that uses Wi-Fi Channel State Information (CSI) and classical machine learning to recognize human activities from changes in the wireless channel.

This repository contains the **Mid-Semester Implementation** for the B.Tech Major Project: "Human and Object Recognition Using WiFi CSI Signals".

## 🎯 Project Objective

The objective of this phase is to build a scientifically correct, end-to-end WiFi CSI Human Activity Recognition system using classical signal processing and machine learning techniques, providing a robust baseline before exploring deep learning models in the final phase.

## 📊 Dataset: UT-HAR

The system uses the **UT-HAR** dataset. Each sample is a multi-dimensional array representing WiFi Channel State Information (CSI):
- **Shape:** `(250, 90)` — 250 time steps × 90 subcarriers.
- **Classes:** 7 human activities.

### Activity Labels
0. Lie down
1. Fall
2. Walk
3. Pick up
4. Run
5. Sit down
6. Stand up

*Note: The dataset was pre-split into `X_train`, `y_train`, `X_test`, `y_test`, `X_val`, `y_val` as NumPy binary files.*

## 🛠️ Environment Setup & Installation

**Prerequisites:**
- Python 3.10+
- `pip`

**Installation:**
```bash
# Clone the repository
git clone https://github.com/ermongroup/Wifi_Activity_Recognition.git
cd Wifi_Activity_Recognition

# Install required packages
pip install -r requirements.txt
```

**Dataset Placement:**
Ensure the dataset files (`X_train.npy`, `y_train.npy`, etc.) are placed inside the `data/` directory at the project root.

## 🚀 Project Architecture & Workflow

The pipeline is designed with strict data leakage prevention.

1.  **Data Loading & Validation:** Loads `.npy` files and verifies shapes and label mappings.
2.  **Preprocessing:** Handles anomalies (NaNs/Infs). Scales features using `StandardScaler` (fitted **only** on the training set).
3.  **Feature Extraction:** Extracts a compact feature vector (~1206 features) per sample using:
    *   **Statistical Features:** Mean, std, variance, RMS, energy, min, max, range, skewness, kurtosis.
    *   **DWT (Discrete Wavelet Transform):** `db4` wavelet on selected subcarriers.
    *   **STFT (Short-Time Fourier Transform):** Spectrogram-based spectral features.
4.  **Dimensionality Reduction (PCA):** Reduces feature space retaining 95% variance (fitted **only** on the training set).
5.  **Model Training:** Trains classical ML classifiers (Random Forest, SVM, KNN).
6.  **Evaluation:** Computes accuracy, precision, recall, F1-score, and confusion matrices.
7.  **Inference (Streamlit):** Loads pre-trained pipeline to predict unseen samples.

## 🏃‍♂️ Training Models

To train the models and generate evaluation metrics and figures, run:

```bash
python -m src.train
```

This will:
- Extract features and train Random Forest, SVM, and KNN.
- Save the trained models, scaler, and PCA objects to `models/`.
- Save performance metrics to `results/metrics/model_metrics.json`.
- Save visualizations (waveforms, spectrograms, confusion matrices) to `results/figures/`.

## 🌐 Running the Streamlit Application

The Streamlit app provides an interactive demonstration of the working ML system on the test set.

**Run the app:**
```bash
streamlit run app.py
```

**Features:**
- Dataset overview and class distributions.
- CSI amplitude visualizations (Heatmap, Waveform, STFT Spectrogram).
- Single-sample inference demonstrating prediction alongside actual labels and confidence probabilities.
- Model performance comparisons.

## 📈 Mid-Semester Scope

**CURRENT IMPLEMENTATION:**
- UT-HAR dataset analysis and visualization.
- Signal processing feature extraction (Statistical, DWT, STFT).
- Dimensionality reduction via PCA.
- Classical Machine Learning models (SVM, Random Forest, KNN).
- Strict data-leakage prevention (fit/transform isolation).
- Streamlit demonstration application.

**FUTURE FINAL IMPLEMENTATION (Post Mid-Sem):**
- Deep Learning Models (CNN, LSTM, CNN-LSTM).
- Advanced feature fusion.
- Object recognition integration.
- Potential real-time CSI data acquisition.

## 📝 Scientific Honesty & Reproducibility

All results presented in this project (accuracy, metrics, figures) are derived from actual dataset execution. A fixed random seed (`42`) is used to ensure reproducible results across runs. Data leakage is explicitly prevented by fitting preprocessing objects exclusively on training data.
