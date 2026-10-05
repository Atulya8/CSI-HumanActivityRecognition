"""
Feature Extraction Module for WiFi CSI Human Activity Recognition.

Implements four feature extraction approaches:
A. Statistical Features — time-domain statistics per subcarrier
B. PCA — dimensionality reduction (fit on training data only)
C. DWT — Discrete Wavelet Transform features
D. STFT — Short-Time Fourier Transform features

All methods extract compact feature vectors suitable for classical ML.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict
from scipy import stats as scipy_stats
from scipy.signal import stft as scipy_stft
import pywt
from sklearn.decomposition import PCA


# Fixed random seed for reproducibility
RANDOM_STATE = 42


# ============================================================
# A. STATISTICAL FEATURES
# ============================================================
# For each of the 90 subcarriers, we compute 10 statistical
# features across the 250 time steps, yielding 90 × 10 = 900
# features per sample.
# ============================================================

STAT_FEATURE_NAMES = [
    "mean", "std", "variance", "rms", "energy",
    "min", "max", "range", "skewness", "kurtosis"
]


def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from 3D CSI data.

    For each sample, computes 10 statistics across the time axis
    for each of the 90 subcarriers.

    Parameters
    ----------
    X : np.ndarray
        CSI data of shape (n_samples, 250, 90)

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_samples, 900)
        (90 subcarriers × 10 statistics)
    """
    n_samples = X.shape[0]
    n_subcarriers = X.shape[2]
    n_features = n_subcarriers * len(STAT_FEATURE_NAMES)

    features = np.zeros((n_samples, n_features), dtype=np.float64)

    for i in range(n_samples):
        sample = X[i]  # (250, 90)
        feat_list = []

        for sc in range(n_subcarriers):
            channel = sample[:, sc]  # (250,) — one subcarrier across time

            mean_val = np.mean(channel)
            std_val = np.std(channel)
            var_val = np.var(channel)
            rms_val = np.sqrt(np.mean(channel ** 2))
            energy_val = np.sum(channel ** 2)
            min_val = np.min(channel)
            max_val = np.max(channel)
            range_val = max_val - min_val
            skew_val = float(scipy_stats.skew(channel))
            kurt_val = float(scipy_stats.kurtosis(channel))

            feat_list.extend([
                mean_val, std_val, var_val, rms_val, energy_val,
                min_val, max_val, range_val, skew_val, kurt_val
            ])

        features[i] = feat_list

    return features


def extract_statistical_features_single(sample: np.ndarray) -> np.ndarray:
    """Extract statistical features from a single sample of shape (250, 90)."""
    if sample.ndim == 2:
        sample = sample[np.newaxis, ...]  # (1, 250, 90)
    return extract_statistical_features(sample)


# ============================================================
# B. PCA (DIMENSIONALITY REDUCTION)
# ============================================================
# PCA is applied AFTER statistical feature extraction to reduce
# the 900-dimensional feature vector. Fit only on training data.
# ============================================================

def fit_pca(
    X_features: np.ndarray,
    n_components: float = 0.95,
    save_path: Optional[Path] = None,
) -> Tuple[PCA, np.ndarray]:
    """
    Fit PCA on training features (NEVER on test data).

    Parameters
    ----------
    X_features : np.ndarray
        Training feature matrix (n_samples, n_features)
    n_components : float
        Explained variance ratio to retain (default: 95%)
    save_path : Path, optional
        Path to save the fitted PCA model

    Returns
    -------
    PCA
        Fitted PCA model
    np.ndarray
        Transformed training features
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_transformed = pca.fit_transform(X_features)

    print(f"  PCA: {X_features.shape[1]} features -> {pca.n_components_} components")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, str(save_path))
        print(f"  PCA model saved to: {save_path}")

    return pca, X_transformed


def transform_pca(X_features: np.ndarray, pca: PCA) -> np.ndarray:
    """Transform features using a pre-fitted PCA model."""
    return pca.transform(X_features)


def load_pca(pca_path: Path) -> PCA:
    """Load a saved PCA model."""
    return joblib.load(str(pca_path))


# ============================================================
# C. DWT (DISCRETE WAVELET TRANSFORM)
# ============================================================
# Uses the Daubechies-4 (db4) wavelet — a common choice for
# biomedical and activity recognition signals due to its good
# time-frequency localization.
#
# Decomposition level: 3 (captures activity-relevant frequency
# bands within the 250-sample window)
#
# Features extracted per subcarrier per decomposition level:
#   - energy, mean, std (3 features × 4 levels = 12 per subcarrier)
#
# For selected subcarriers to keep feature count manageable.
# ============================================================

DWT_WAVELET = "db4"
DWT_LEVEL = 3
# Select every 5th subcarrier (18 out of 90) to keep features compact
DWT_SUBCARRIER_STEP = 5
DWT_FEATURES_PER_COEFF = 3  # energy, mean, std


def extract_dwt_features(X: np.ndarray) -> np.ndarray:
    """
    Extract DWT features from 3D CSI data.

    Uses db4 wavelet with 3 levels of decomposition on selected
    subcarriers. Extracts energy, mean, std from each coefficient
    set (cA3, cD3, cD2, cD1 = 4 coefficient sets).

    Parameters
    ----------
    X : np.ndarray
        CSI data of shape (n_samples, 250, 90)

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_samples, n_dwt_features)
    """
    n_samples = X.shape[0]
    n_subcarriers = X.shape[2]
    selected_subcarriers = list(range(0, n_subcarriers, DWT_SUBCARRIER_STEP))
    n_selected = len(selected_subcarriers)

    # 4 coefficient sets (cA3 + cD3 + cD2 + cD1) × 3 features each × n_selected subcarriers
    n_coeff_sets = DWT_LEVEL + 1
    n_features = n_selected * n_coeff_sets * DWT_FEATURES_PER_COEFF

    features = np.zeros((n_samples, n_features), dtype=np.float64)

    for i in range(n_samples):
        feat_idx = 0
        for sc_idx in selected_subcarriers:
            channel = X[i, :, sc_idx]  # (250,)

            # Perform DWT decomposition
            coeffs = pywt.wavedec(channel, DWT_WAVELET, level=DWT_LEVEL)
            # coeffs = [cA3, cD3, cD2, cD1]

            for coeff in coeffs:
                energy = np.sum(coeff ** 2)
                mean_val = np.mean(coeff)
                std_val = np.std(coeff)
                features[i, feat_idx] = energy
                features[i, feat_idx + 1] = mean_val
                features[i, feat_idx + 2] = std_val
                feat_idx += 3

    return features


def extract_dwt_features_single(sample: np.ndarray) -> np.ndarray:
    """Extract DWT features from a single sample of shape (250, 90)."""
    if sample.ndim == 2:
        sample = sample[np.newaxis, ...]
    return extract_dwt_features(sample)


# ============================================================
# D. STFT (SHORT-TIME FOURIER TRANSFORM)
# ============================================================
# STFT captures time-frequency characteristics of the CSI signal.
#
# NOTE: The UT-HAR dataset does not specify a physical sampling
# frequency. We use a normalized frequency axis (samples as the
# unit). The STFT parameters are set relative to the 250-sample
# window length.
#
# Parameters:
#   - Window size (nperseg): 32 samples
#   - Overlap (noverlap): 16 samples (50%)
#   - FFT points: 32
#   - fs: 1.0 (normalized, sample-index based)
#
# Features extracted per selected subcarrier:
#   - mean spectral energy per frequency bin
#   - max spectral energy
#   - spectral centroid
#   - spectral bandwidth
#   - total energy
# (5 features per selected subcarrier)
# ============================================================

STFT_NPERSEG = 32
STFT_NOVERLAP = 16
STFT_FS = 1.0  # Normalized — no physical sampling frequency available
STFT_SUBCARRIER_STEP = 5  # Same selection as DWT
STFT_FEATURES_PER_SC = 5


def extract_stft_features(X: np.ndarray) -> np.ndarray:
    """
    Extract STFT-based features from 3D CSI data.

    Computes the STFT for selected subcarriers and extracts
    compact spectral features from the magnitude spectrogram.

    Parameters
    ----------
    X : np.ndarray
        CSI data of shape (n_samples, 250, 90)

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_samples, n_stft_features)
    """
    n_samples = X.shape[0]
    n_subcarriers = X.shape[2]
    selected_subcarriers = list(range(0, n_subcarriers, STFT_SUBCARRIER_STEP))
    n_selected = len(selected_subcarriers)
    n_features = n_selected * STFT_FEATURES_PER_SC

    features = np.zeros((n_samples, n_features), dtype=np.float64)

    for i in range(n_samples):
        feat_idx = 0
        for sc_idx in selected_subcarriers:
            channel = X[i, :, sc_idx]  # (250,)

            # Compute STFT
            freqs, times, Zxx = scipy_stft(
                channel,
                fs=STFT_FS,
                nperseg=STFT_NPERSEG,
                noverlap=STFT_NOVERLAP,
            )

            # Magnitude spectrogram
            magnitude = np.abs(Zxx)  # (n_freqs, n_times)

            # Feature 1: Mean spectral energy across time
            mean_energy = np.mean(magnitude ** 2)

            # Feature 2: Max spectral energy
            max_energy = np.max(magnitude ** 2)

            # Feature 3: Spectral centroid (weighted mean frequency)
            total_mag = np.sum(magnitude)
            if total_mag > 0:
                spectral_centroid = np.sum(freqs[:, np.newaxis] * magnitude) / total_mag
            else:
                spectral_centroid = 0.0

            # Feature 4: Spectral bandwidth
            if total_mag > 0:
                spectral_bw = np.sqrt(
                    np.sum(((freqs[:, np.newaxis] - spectral_centroid) ** 2) * magnitude) / total_mag
                )
            else:
                spectral_bw = 0.0

            # Feature 5: Total energy
            total_energy = np.sum(magnitude ** 2)

            features[i, feat_idx] = mean_energy
            features[i, feat_idx + 1] = max_energy
            features[i, feat_idx + 2] = spectral_centroid
            features[i, feat_idx + 3] = spectral_bw
            features[i, feat_idx + 4] = total_energy
            feat_idx += 5

    return features


def extract_stft_features_single(sample: np.ndarray) -> np.ndarray:
    """Extract STFT features from a single sample of shape (250, 90)."""
    if sample.ndim == 2:
        sample = sample[np.newaxis, ...]
    return extract_stft_features(sample)


# ============================================================
# COMBINED FEATURE EXTRACTION
# ============================================================

def extract_all_features(
    X: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract and concatenate all feature types.

    Feature vector composition:
    - Statistical: 90 subcarriers × 10 stats = 900 features
    - DWT: 18 subcarriers × 4 coeff sets × 3 stats = 216 features
    - STFT: 18 subcarriers × 5 spectral features = 90 features
    - Total: ~1206 features (before PCA)

    Parameters
    ----------
    X : np.ndarray
        CSI data of shape (n_samples, 250, 90)
    verbose : bool
        Print progress information

    Returns
    -------
    np.ndarray
        Combined feature matrix
    """
    if verbose:
        print(f"\n  Extracting features from {X.shape[0]} samples...")

    # A. Statistical features
    if verbose:
        print("  [A] Statistical features...")
    stat_features = extract_statistical_features(X)
    if verbose:
        print(f"      -> {stat_features.shape[1]} features")

    # C. DWT features
    if verbose:
        print("  [C] DWT features...")
    dwt_features = extract_dwt_features(X)
    if verbose:
        print(f"      -> {dwt_features.shape[1]} features")

    # D. STFT features
    if verbose:
        print("  [D] STFT features...")
    stft_features = extract_stft_features(X)
    if verbose:
        print(f"      -> {stft_features.shape[1]} features")

    # Concatenate all features
    combined = np.hstack([stat_features, dwt_features, stft_features])

    if verbose:
        print(f"  Combined feature vector: {combined.shape[1]} features per sample")

    return combined


def extract_all_features_single(sample: np.ndarray) -> np.ndarray:
    """Extract all features from a single sample of shape (250, 90)."""
    if sample.ndim == 2:
        sample = sample[np.newaxis, ...]
    return extract_all_features(sample, verbose=False)


def get_feature_info() -> Dict:
    """Return information about the feature extraction configuration."""
    n_subcarriers = 90
    n_selected = len(range(0, n_subcarriers, DWT_SUBCARRIER_STEP))

    return {
        "statistical": {
            "description": "Time-domain statistics per subcarrier",
            "features_per_subcarrier": len(STAT_FEATURE_NAMES),
            "num_subcarriers": n_subcarriers,
            "total": n_subcarriers * len(STAT_FEATURE_NAMES),
            "feature_names": STAT_FEATURE_NAMES,
        },
        "dwt": {
            "description": f"Wavelet coefficients ({DWT_WAVELET}, level={DWT_LEVEL})",
            "wavelet": DWT_WAVELET,
            "level": DWT_LEVEL,
            "subcarrier_step": DWT_SUBCARRIER_STEP,
            "selected_subcarriers": n_selected,
            "features_per_subcarrier": (DWT_LEVEL + 1) * DWT_FEATURES_PER_COEFF,
            "total": n_selected * (DWT_LEVEL + 1) * DWT_FEATURES_PER_COEFF,
        },
        "stft": {
            "description": "Short-Time Fourier Transform spectral features",
            "nperseg": STFT_NPERSEG,
            "noverlap": STFT_NOVERLAP,
            "fs": f"{STFT_FS} (normalized, no physical sampling frequency available)",
            "subcarrier_step": STFT_SUBCARRIER_STEP,
            "selected_subcarriers": n_selected,
            "features_per_subcarrier": STFT_FEATURES_PER_SC,
            "total": n_selected * STFT_FEATURES_PER_SC,
        },
    }
