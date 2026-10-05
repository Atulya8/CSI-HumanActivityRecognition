"""
Model Training Module for WiFi CSI Human Activity Recognition.

Trains classical ML models (Random Forest, SVM, KNN) on extracted
CSI features. Implements the complete training pipeline:

1. Load dataset
2. Extract features (Statistical + DWT + STFT)
3. Apply StandardScaler (fit on train only)
4. Apply PCA (fit on train only)
5. Train models
6. Evaluate on test set
7. Save all artifacts

Data leakage prevention:
- Scaler: fit on training features, transform test features
- PCA: fit on training features, transform test features
- Models: trained on training data only
"""

import numpy as np
import joblib
import json
import time
from pathlib import Path
from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset, validate_dataset, ACTIVITY_LABELS, get_data_dir
from src.feature_extraction import (
    extract_all_features,
    fit_pca,
    transform_pca,
    get_feature_info,
)
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pca_variance,
    plot_csi_waveform,
    plot_csi_heatmap,
    plot_spectrogram,
)

# Fixed random seed for reproducibility
RANDOM_STATE = 42


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def train_pipeline():
    """
    Complete training pipeline.

    This is the main entry point for training all models.
    Run this script to train models before using the Streamlit app.
    """
    project_root = get_project_root()
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"

    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WiFi CSI Human Activity Recognition — Training Pipeline")
    print("=" * 60)

    # ============================================================
    # Step 1: Load and validate dataset
    # ============================================================
    print("\n[STEP 1] Loading dataset...")
    dataset = load_dataset()
    validate_dataset(dataset)

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Classes: {len(ACTIVITY_LABELS)}")

    # ============================================================
    # Step 2: Generate dataset visualizations
    # ============================================================
    print("\n[STEP 2] Generating dataset visualizations...")

    # Class distribution
    plot_class_distribution(
        y_train,
        title="Training Set — Class Distribution",
        save_path=figures_dir / "class_distribution_train.png",
    )
    plot_class_distribution(
        y_test,
        title="Test Set — Class Distribution",
        save_path=figures_dir / "class_distribution_test.png",
    )

    # Sample CSI visualizations (one per activity)
    for label, name in ACTIVITY_LABELS.items():
        # Find first sample of this activity
        idx = np.where(y_train == label)[0][0]
        sample = X_train[idx]

        plot_csi_waveform(
            sample,
            activity_name=name,
            save_path=figures_dir / f"csi_waveform_{name.lower().replace(' ', '_')}.png",
        )
        plot_csi_heatmap(
            sample,
            activity_name=name,
            save_path=figures_dir / f"csi_heatmap_{name.lower().replace(' ', '_')}.png",
        )
        plot_spectrogram(
            sample,
            subcarrier=0,
            activity_name=name,
            save_path=figures_dir / f"spectrogram_{name.lower().replace(' ', '_')}.png",
        )

    # ============================================================
    # Step 3: Feature extraction
    # ============================================================
    print("\n[STEP 3] Extracting features...")
    t0 = time.time()
    
    features_cache_path = metrics_dir / "features_cache.npz"
    if features_cache_path.exists():
        print("  Loading features from cache...")
        cache = np.load(str(features_cache_path))
        X_train_features = cache["X_train_features"]
        X_test_features = cache["X_test_features"]
    else:
        print("\n  Training set:")
        X_train_features = extract_all_features(X_train)

        print("\n  Test set:")
        X_test_features = extract_all_features(X_test)
        
        # Save to cache
        np.savez(
            str(features_cache_path), 
            X_train_features=X_train_features, 
            X_test_features=X_test_features
        )
        print(f"\n  Features cached to {features_cache_path}")

    feat_time = time.time() - t0
    print(f"\n  Feature extraction time: {feat_time:.1f}s")
    print(f"  Training features shape: {X_train_features.shape}")
    print(f"  Test features shape: {X_test_features.shape}")

    # Save feature info
    feat_info = get_feature_info()
    with open(metrics_dir / "feature_info.json", "w") as f:
        json.dump(feat_info, f, indent=2)

    # ============================================================
    # Step 4: Scaling (fit on train only)
    # ============================================================
    print("\n[STEP 4] Scaling features...")

    # Handle any NaN/Inf from feature extraction
    X_train_features = np.nan_to_num(X_train_features, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_features = np.nan_to_num(X_test_features, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)  # FIT on train
    X_test_scaled = scaler.transform(X_test_features)  # TRANSFORM only on test

    # Save scaler
    joblib.dump(scaler, str(models_dir / "scaler.pkl"))
    print(f"  Scaler saved to: {models_dir / 'scaler.pkl'}")

    # ============================================================
    # Step 5: PCA (fit on train only)
    # ============================================================
    print("\n[STEP 5] Applying PCA...")

    pca, X_train_pca = fit_pca(
        X_train_scaled,
        n_components=0.95,
        save_path=models_dir / "pca.pkl",
    )
    X_test_pca = transform_pca(X_test_scaled, pca)

    print(f"  Training PCA shape: {X_train_pca.shape}")
    print(f"  Test PCA shape: {X_test_pca.shape}")

    # Save PCA variance plot
    plot_pca_variance(
        pca.explained_variance_ratio_,
        save_path=figures_dir / "pca_explained_variance.png",
    )

    # ============================================================
    # Step 6: Train models
    # ============================================================
    print("\n[STEP 6] Training models...")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,  # Enable probability estimates
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            n_jobs=-1,
        ),
    }

    all_metrics = {}

    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        t0 = time.time()

        model.fit(X_train_pca, y_train)
        train_time = time.time() - t0
        print(f"    Training time: {train_time:.2f}s")

        # Predict on test set
        y_pred = model.predict(X_test_pca)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-score:  {f1:.4f}")

        all_metrics[model_name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "training_time": round(train_time, 2),
        }

        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=[ACTIVITY_LABELS[i] for i in range(len(ACTIVITY_LABELS))],
            zero_division=0,
        )
        print(f"\n    Classification Report:\n{report}")

        # Save classification report
        report_path = metrics_dir / f"classification_report_{model_name.lower().replace(' ', '_')}.txt"
        with open(report_path, "w") as f:
            f.write(f"Classification Report — {model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)

        # Save confusion matrix figure
        plot_confusion_matrix(
            y_test, y_pred,
            model_name=model_name,
            save_path=figures_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
        )

        # Save model
        model_filename = f"{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, str(models_dir / model_filename))
        print(f"    Model saved to: {models_dir / model_filename}")

    # ============================================================
    # Step 7: Save comparison results
    # ============================================================
    print("\n[STEP 7] Saving comparison results...")

    # Model comparison chart
    plot_model_comparison(
        all_metrics,
        save_path=figures_dir / "model_comparison.png",
    )

    # Save metrics JSON
    with open(metrics_dir / "model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_dir / 'model_metrics.json'}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for model_name, metrics in all_metrics.items():
        print(
            f"{model_name:<20} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f}"
        )
    print("=" * 60)

    print("\n✓ Training pipeline complete!")
    print(f"  Models saved in: {models_dir}")
    print(f"  Figures saved in: {figures_dir}")
    print(f"  Metrics saved in: {metrics_dir}")

    return all_metrics


if __name__ == "__main__":
    train_pipeline()
