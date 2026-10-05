"""
Preprocessing Module for WiFi CSI Human Activity Recognition.

Implements a reproducible preprocessing pipeline with strict
data leakage prevention:
  - StandardScaler is fit ONLY on training data
  - All transformations on test/val data use training-fitted parameters
  - Fitted objects are saved via joblib for reuse in the Streamlit app
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


# Fixed random seed for reproducibility
RANDOM_STATE = 42


def check_data_quality(X: np.ndarray, name: str = "X") -> np.ndarray:
    """
    Check and clean data quality issues (NaN, Inf).

    Parameters
    ----------
    X : np.ndarray
        Input data array
    name : str
        Name for logging

    Returns
    -------
    np.ndarray
        Cleaned data array
    """
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    if nan_count > 0:
        print(f"  WARNING: {name} contains {nan_count} NaN values. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0)

    if inf_count > 0:
        print(f"  WARNING: {name} contains {inf_count} Inf values. Replacing with finite bounds.")
        X = np.nan_to_num(X, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

    print(f"  {name}: shape={X.shape}, range=[{X.min():.4f}, {X.max():.4f}]")
    return X


def flatten_csi(X: np.ndarray) -> np.ndarray:
    """
    Flatten 3D CSI data (samples, timesteps, subcarriers) to 2D (samples, features).

    This is used when applying StandardScaler across the full feature space.
    Each sample of shape (250, 90) becomes a single vector of 22500 features.

    Parameters
    ----------
    X : np.ndarray
        3D array of shape (n_samples, n_timesteps, n_subcarriers)

    Returns
    -------
    np.ndarray
        2D array of shape (n_samples, n_timesteps * n_subcarriers)
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def create_scaler(X_train_flat: np.ndarray, save_path: Optional[Path] = None) -> StandardScaler:
    """
    Create and fit a StandardScaler on training data ONLY.

    IMPORTANT: This scaler must NEVER be fit on test or validation data.

    Parameters
    ----------
    X_train_flat : np.ndarray
        Flattened training data of shape (n_samples, n_features)
    save_path : Path, optional
        Path to save the fitted scaler

    Returns
    -------
    StandardScaler
        Fitted scaler
    """
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    print(f"  Scaler fit on training data: {X_train_flat.shape}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, str(save_path))
        print(f"  Scaler saved to: {save_path}")

    return scaler


def preprocess_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
) -> dict:
    """
    Complete preprocessing pipeline.

    Steps:
    1. Check data quality (NaN/Inf)
    2. Flatten CSI data for scaling
    3. Fit StandardScaler on training data ONLY
    4. Transform all splits using the training-fitted scaler

    Parameters
    ----------
    X_train : np.ndarray
        Training data (n_samples, 250, 90)
    X_test : np.ndarray
        Test data
    X_val : np.ndarray, optional
        Validation data
    save_dir : Path, optional
        Directory to save fitted preprocessing objects

    Returns
    -------
    dict
        Preprocessed data and fitted objects
    """
    print("\n--- Preprocessing Pipeline ---")

    # Step 1: Data quality check
    print("\n[1] Checking data quality...")
    X_train = check_data_quality(X_train, "X_train")
    X_test = check_data_quality(X_test, "X_test")
    if X_val is not None:
        X_val = check_data_quality(X_val, "X_val")

    # Store original 3D shapes for feature extraction
    # Feature extraction works on the 3D data (samples, timesteps, subcarriers)
    result = {
        "X_train_3d": X_train.copy(),
        "X_test_3d": X_test.copy(),
    }
    if X_val is not None:
        result["X_val_3d"] = X_val.copy()

    # Step 2: Flatten for scaling
    print("\n[2] Flattening CSI data...")
    X_train_flat = flatten_csi(X_train)
    X_test_flat = flatten_csi(X_test)
    print(f"  X_train_flat: {X_train_flat.shape}")
    print(f"  X_test_flat: {X_test_flat.shape}")

    if X_val is not None:
        X_val_flat = flatten_csi(X_val)
        print(f"  X_val_flat: {X_val_flat.shape}")

    # Step 3: Fit scaler on TRAINING data only
    print("\n[3] Fitting StandardScaler on training data only...")
    scaler_path = Path(save_dir) / "scaler.pkl" if save_dir else None
    scaler = create_scaler(X_train_flat, save_path=scaler_path)

    # Step 4: Transform all splits using the training-fitted scaler
    print("\n[4] Transforming data...")
    result["X_train_scaled"] = scaler.transform(X_train_flat)
    result["X_test_scaled"] = scaler.transform(X_test_flat)
    print(f"  X_train_scaled: range=[{result['X_train_scaled'].min():.4f}, {result['X_train_scaled'].max():.4f}]")
    print(f"  X_test_scaled: range=[{result['X_test_scaled'].min():.4f}, {result['X_test_scaled'].max():.4f}]")

    if X_val is not None:
        result["X_val_scaled"] = scaler.transform(X_val_flat)

    result["scaler"] = scaler

    print("\n--- Preprocessing Complete ---")
    return result


def load_scaler(scaler_path: Path) -> StandardScaler:
    """Load a saved StandardScaler."""
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    return joblib.load(str(scaler_path))


def preprocess_single_sample(
    sample: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Preprocess a single CSI sample for prediction.

    Uses the SAVED scaler (fitted on training data) to transform
    the sample — no new fitting occurs.

    Parameters
    ----------
    sample : np.ndarray
        Single CSI sample of shape (250, 90)
    scaler : StandardScaler
        Pre-fitted scaler from training

    Returns
    -------
    np.ndarray
        Preprocessed sample of shape (1, 22500)
    """
    if sample.ndim == 2:
        sample = sample.reshape(1, -1)  # (1, 22500)
    elif sample.ndim == 3:
        sample = sample.reshape(sample.shape[0], -1)

    return scaler.transform(sample)
