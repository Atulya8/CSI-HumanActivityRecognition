"""
Data Loader Module for UT-HAR WiFi CSI Dataset.

Handles loading, validation, and basic inspection of the UT-HAR dataset.
The dataset contains WiFi Channel State Information (CSI) data for
7 human activities: Lie down, Fall, Walk, Pick up, Run, Sit down, Stand up.

Each sample has shape (250, 90):
  - 250 time steps (temporal samples in the CSI window)
  - 90 subcarrier features (CSI amplitude values across OFDM subcarriers)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# ============================================================
# LABEL MAPPING
# ============================================================
# Verified from UT-HAR literature and dataset documentation.
# The UT-HAR dataset uses 7 activity classes labeled 0-6.
# This mapping is consistent with the original UT-HAR paper
# and the SenseFi benchmark.
# ============================================================
ACTIVITY_LABELS: Dict[int, str] = {
    0: "Lie down",
    1: "Fall",
    2: "Walk",
    3: "Pick up",
    4: "Run",
    5: "Sit down",
    6: "Stand up",
}

NUM_CLASSES = 7
NUM_TIMESTEPS = 250
NUM_SUBCARRIERS = 90


def get_data_dir() -> Path:
    """Return the path to the data directory relative to project root."""
    # Works whether called from project root or from src/
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found at {data_dir}. "
            "Please ensure the UT-HAR dataset .npy files are in the 'data/' directory."
        )
    return data_dir


def load_dataset(data_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """
    Load the complete UT-HAR dataset.

    Parameters
    ----------
    data_dir : Path, optional
        Path to the data directory. If None, auto-detected.

    Returns
    -------
    dict
        Dictionary with keys: X_train, y_train, X_test, y_test, X_val, y_val
    """
    if data_dir is None:
        data_dir = get_data_dir()
    else:
        data_dir = Path(data_dir)

    required_files = {
        "X_train": "X_train.npy",
        "y_train": "y_train.npy",
        "X_test": "X_test.npy",
        "y_test": "y_test.npy",
    }

    optional_files = {
        "X_val": "X_val.npy",
        "y_val": "y_val.npy",
    }

    dataset = {}

    # Load required files
    for key, filename in required_files.items():
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {filepath}\n"
                f"Please place the UT-HAR .npy files in: {data_dir}"
            )
        dataset[key] = np.load(str(filepath), allow_pickle=True)
        print(f"  Loaded {key}: shape={dataset[key].shape}, dtype={dataset[key].dtype}")

    # Load optional files
    for key, filename in optional_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            dataset[key] = np.load(str(filepath), allow_pickle=True)
            print(f"  Loaded {key}: shape={dataset[key].shape}, dtype={dataset[key].dtype}")
        else:
            print(f"  Optional file {filename} not found, skipping.")

    return dataset


def validate_dataset(dataset: Dict[str, np.ndarray]) -> bool:
    """
    Validate the loaded dataset for consistency.

    Checks:
    1. X and y lengths match
    2. X has expected 3D shape (samples, timesteps, subcarriers)
    3. Labels are in expected range [0, 6]
    4. No NaN or Inf values in X

    Parameters
    ----------
    dataset : dict
        Dataset dictionary from load_dataset()

    Returns
    -------
    bool
        True if all validations pass

    Raises
    ------
    ValueError
        If any validation fails
    """
    errors = []

    # Check X-y length consistency
    for split in ["train", "test", "val"]:
        x_key = f"X_{split}"
        y_key = f"y_{split}"
        if x_key in dataset and y_key in dataset:
            if len(dataset[x_key]) != len(dataset[y_key]):
                errors.append(
                    f"Length mismatch: {x_key} has {len(dataset[x_key])} samples "
                    f"but {y_key} has {len(dataset[y_key])} samples."
                )

    # Check X shape (should be 3D: samples × timesteps × subcarriers)
    for key in ["X_train", "X_test", "X_val"]:
        if key in dataset:
            if dataset[key].ndim != 3:
                errors.append(
                    f"{key} should be 3D (samples, timesteps, subcarriers), "
                    f"got {dataset[key].ndim}D with shape {dataset[key].shape}"
                )
            else:
                _, t, s = dataset[key].shape
                if t != NUM_TIMESTEPS or s != NUM_SUBCARRIERS:
                    print(
                        f"  Warning: {key} has shape (*, {t}, {s}), "
                        f"expected (*, {NUM_TIMESTEPS}, {NUM_SUBCARRIERS})"
                    )

    # Check labels are in valid range
    for key in ["y_train", "y_test", "y_val"]:
        if key in dataset:
            unique_labels = np.unique(dataset[key])
            invalid = [l for l in unique_labels if l not in ACTIVITY_LABELS]
            if invalid:
                errors.append(
                    f"{key} contains invalid labels: {invalid}. "
                    f"Expected labels: {list(ACTIVITY_LABELS.keys())}"
                )

    # Check for NaN/Inf
    for key in ["X_train", "X_test", "X_val"]:
        if key in dataset:
            if np.isnan(dataset[key]).any():
                errors.append(f"{key} contains NaN values.")
            if np.isinf(dataset[key]).any():
                errors.append(f"{key} contains Inf values.")

    if errors:
        error_msg = "Dataset validation FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    print("  Dataset validation PASSED — all checks OK.")
    return True


def get_dataset_summary(dataset: Dict[str, np.ndarray]) -> Dict:
    """
    Generate a comprehensive summary of the dataset.

    Returns
    -------
    dict
        Dictionary containing dataset statistics
    """
    summary = {
        "num_classes": NUM_CLASSES,
        "activity_labels": ACTIVITY_LABELS,
        "num_timesteps": NUM_TIMESTEPS,
        "num_subcarriers": NUM_SUBCARRIERS,
    }

    for split in ["train", "test", "val"]:
        x_key = f"X_{split}"
        y_key = f"y_{split}"
        if x_key in dataset:
            X = dataset[x_key]
            summary[f"{split}_samples"] = len(X)
            summary[f"{split}_shape"] = X.shape
            summary[f"{split}_dtype"] = str(X.dtype)
            summary[f"{split}_min"] = float(np.min(X))
            summary[f"{split}_max"] = float(np.max(X))
            summary[f"{split}_mean"] = float(np.mean(X))
            summary[f"{split}_std"] = float(np.std(X))

        if y_key in dataset:
            y = dataset[y_key]
            unique, counts = np.unique(y, return_counts=True)
            summary[f"{split}_class_distribution"] = {
                ACTIVITY_LABELS[int(u)]: int(c) for u, c in zip(unique, counts)
            }

    return summary


def get_activity_name(label: int) -> str:
    """Get activity name from numeric label."""
    if label not in ACTIVITY_LABELS:
        raise ValueError(f"Unknown label: {label}. Valid labels: {list(ACTIVITY_LABELS.keys())}")
    return ACTIVITY_LABELS[label]


def get_label_from_name(name: str) -> int:
    """Get numeric label from activity name."""
    for label, activity in ACTIVITY_LABELS.items():
        if activity.lower() == name.lower():
            return label
    raise ValueError(f"Unknown activity: {name}. Valid activities: {list(ACTIVITY_LABELS.values())}")


if __name__ == "__main__":
    print("=" * 60)
    print("UT-HAR Dataset Loader")
    print("=" * 60)
    print("\nLoading dataset...")
    dataset = load_dataset()
    print("\nValidating dataset...")
    validate_dataset(dataset)
    print("\nDataset Summary:")
    summary = get_dataset_summary(dataset)
    for key, value in summary.items():
        print(f"  {key}: {value}")
