"""
Prediction Module for WiFi CSI Human Activity Recognition.

Loads pre-trained models and preprocessing pipeline to predict
human activities from unseen CSI samples.

Pipeline:
  CSI Sample (250, 90)
      ↓
  Feature Extraction (Statistical + DWT + STFT)
      ↓
  Scaler.transform() (pre-fitted on training data)
      ↓
  PCA.transform() (pre-fitted on training data)
      ↓
  Model.predict() / predict_proba()
      ↓
  Activity Prediction + Probabilities
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.data_loader import ACTIVITY_LABELS, NUM_TIMESTEPS, NUM_SUBCARRIERS
from src.feature_extraction import extract_all_features_single


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_pipeline(models_dir: Optional[Path] = None) -> Dict:
    """
    Load the complete prediction pipeline (scaler, PCA, models).

    Parameters
    ----------
    models_dir : Path, optional
        Directory containing saved models. If None, auto-detected.

    Returns
    -------
    dict
        Dictionary containing: scaler, pca, and model objects
    """
    if models_dir is None:
        models_dir = get_project_root() / "models"

    models_dir = Path(models_dir)

    pipeline = {}

    # Load scaler
    scaler_path = models_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Please run the training pipeline first (python -m src.train)."
        )
    pipeline["scaler"] = joblib.load(str(scaler_path))

    # Load PCA
    pca_path = models_dir / "pca.pkl"
    if not pca_path.exists():
        raise FileNotFoundError(
            f"PCA model not found at {pca_path}. "
            "Please run the training pipeline first."
        )
    pipeline["pca"] = joblib.load(str(pca_path))

    # Load available models
    pipeline["models"] = {}
    model_files = {
        "Random Forest": "random_forest.pkl",
        "SVM": "svm.pkl",
        "KNN": "knn.pkl",
    }

    for model_name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            pipeline["models"][model_name] = joblib.load(str(model_path))
            print(f"  Loaded model: {model_name}")
        else:
            print(f"  Model not found: {model_name} ({model_path})")

    if not pipeline["models"]:
        raise FileNotFoundError(
            "No trained models found. Please run the training pipeline first."
        )

    return pipeline


def predict_sample(
    sample: np.ndarray,
    pipeline: Dict,
    model_name: str = "Random Forest",
) -> Dict:
    """
    Predict the activity for a single CSI sample.

    The ground-truth label is NOT used during prediction.
    It should only be compared AFTER the prediction is made.

    Parameters
    ----------
    sample : np.ndarray
        Single CSI sample of shape (250, 90)
    pipeline : dict
        Loaded pipeline from load_pipeline()
    model_name : str
        Which model to use for prediction

    Returns
    -------
    dict
        Prediction results including:
        - predicted_label: int
        - predicted_activity: str
        - probabilities: np.ndarray (probability per class)
        - confidence: float (probability of predicted class)
    """
    # Validate input shape
    if sample.ndim != 2 or sample.shape != (NUM_TIMESTEPS, NUM_SUBCARRIERS):
        raise ValueError(
            f"Expected sample shape ({NUM_TIMESTEPS}, {NUM_SUBCARRIERS}), "
            f"got {sample.shape}"
        )

    if model_name not in pipeline["models"]:
        available = list(pipeline["models"].keys())
        raise ValueError(
            f"Model '{model_name}' not available. Available: {available}"
        )

    model = pipeline["models"][model_name]
    scaler = pipeline["scaler"]
    pca = pipeline["pca"]

    # Step 1: Feature extraction (same as training)
    features = extract_all_features_single(sample)  # (1, n_features)

    # Handle any NaN/Inf from feature extraction
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 2: Scale using SAVED scaler (no re-fitting!)
    features_scaled = scaler.transform(features)

    # Step 3: PCA using SAVED model (no re-fitting!)
    features_pca = pca.transform(features_scaled)

    # Step 4: Predict
    predicted_label = int(model.predict(features_pca)[0])

    # Step 5: Get probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_pca)[0]
    else:
        # Fallback: one-hot for models without predict_proba
        probabilities = np.zeros(len(ACTIVITY_LABELS))
        probabilities[predicted_label] = 1.0

    confidence = float(probabilities[predicted_label])

    return {
        "predicted_label": predicted_label,
        "predicted_activity": ACTIVITY_LABELS[predicted_label],
        "probabilities": probabilities,
        "confidence": confidence,
        "all_activities": {
            ACTIVITY_LABELS[i]: float(probabilities[i])
            for i in range(len(ACTIVITY_LABELS))
        },
    }


def batch_predict(
    X: np.ndarray,
    pipeline: Dict,
    model_name: str = "Random Forest",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict activities for a batch of CSI samples.

    Parameters
    ----------
    X : np.ndarray
        CSI data of shape (n_samples, 250, 90)
    pipeline : dict
        Loaded pipeline from load_pipeline()
    model_name : str
        Which model to use

    Returns
    -------
    y_pred : np.ndarray
        Predicted labels
    probabilities : np.ndarray
        Probability matrix of shape (n_samples, n_classes)
    """
    from src.feature_extraction import extract_all_features

    model = pipeline["models"][model_name]
    scaler = pipeline["scaler"]
    pca = pipeline["pca"]

    # Feature extraction
    features = extract_all_features(X, verbose=True)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale and PCA (using saved objects — no re-fitting)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)

    # Predict
    y_pred = model.predict(features_pca)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_pca)
    else:
        probabilities = np.zeros((len(y_pred), len(ACTIVITY_LABELS)))
        for i, label in enumerate(y_pred):
            probabilities[i, label] = 1.0

    return y_pred, probabilities


if __name__ == "__main__":
    # Quick demo: predict a single test sample
    print("=" * 60)
    print("Prediction Demo")
    print("=" * 60)

    # Load test data
    data_dir = get_project_root() / "data"
    X_test = np.load(str(data_dir / "X_test.npy"), allow_pickle=True)
    y_test = np.load(str(data_dir / "y_test.npy"), allow_pickle=True)

    # Load pipeline
    print("\nLoading pipeline...")
    pipeline = load_pipeline()

    # Predict sample 0
    sample_idx = 25
    sample = X_test[sample_idx]
    actual_label = int(y_test[sample_idx])

    print(f"\nPredicting test sample {sample_idx}...")
    result = predict_sample(sample, pipeline, model_name="Random Forest")

    print(f"\n{'─' * 40}")
    print(f"  Predicted: {result['predicted_activity']}")
    print(f"  Actual:    {ACTIVITY_LABELS[actual_label]}")
    print(f"  Confidence: {result['confidence'] * 100:.1f}%")
    print(f"{'─' * 40}")
    print("\n  Class Probabilities:")
    for activity, prob in result["all_activities"].items():
        marker = " ◀" if activity == result["predicted_activity"] else ""
        print(f"    {activity:<12} {prob * 100:6.2f}%{marker}")
