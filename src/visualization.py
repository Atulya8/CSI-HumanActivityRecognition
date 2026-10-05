"""
Visualization Module for WiFi CSI Human Activity Recognition.

Provides functions to generate publication-quality visualizations:
- Raw CSI waveforms
- CSI amplitude variation over time
- Spectrograms (STFT)
- Class distribution charts
- Confusion matrices
- Model comparison charts
- PCA explained variance plots
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List
from scipy.signal import stft as scipy_stft
from sklearn.metrics import confusion_matrix

from src.data_loader import ACTIVITY_LABELS
from src.feature_extraction import STFT_FS, STFT_NPERSEG, STFT_NOVERLAP

# Plot style configuration
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_class_distribution(
    y: np.ndarray,
    title: str = "Class Distribution",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot the distribution of activity classes.

    Parameters
    ----------
    y : np.ndarray
        Label array
    title : str
        Plot title
    save_path : Path, optional
        Path to save the figure
    """
    unique, counts = np.unique(y, return_counts=True)
    labels = [ACTIVITY_LABELS[int(u)] for u in unique]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("husl", len(unique))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_xlabel("Activity")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_csi_waveform(
    sample: np.ndarray,
    activity_name: str = "",
    subcarriers: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot raw CSI waveform for selected subcarriers.

    Parameters
    ----------
    sample : np.ndarray
        Single CSI sample of shape (250, 90)
    activity_name : str
        Activity label for title
    subcarriers : list of int, optional
        Subcarrier indices to plot. Default: 6 evenly spaced.
    save_path : Path, optional
        Path to save the figure
    """
    if subcarriers is None:
        subcarriers = [0, 15, 30, 45, 60, 75]

    fig, axes = plt.subplots(len(subcarriers), 1, figsize=(12, 2.5 * len(subcarriers)), sharex=True)
    if len(subcarriers) == 1:
        axes = [axes]

    colors = sns.color_palette("viridis", len(subcarriers))
    time_axis = np.arange(sample.shape[0])

    for idx, (sc, ax) in enumerate(zip(subcarriers, axes)):
        ax.plot(time_axis, sample[:, sc], color=colors[idx], linewidth=0.8)
        ax.set_ylabel(f"SC {sc}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step (sample index)")
    title = "CSI Waveform — Amplitude vs. Time"
    if activity_name:
        title += f" [{activity_name}]"
    axes[0].set_title(title)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_csi_heatmap(
    sample: np.ndarray,
    activity_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot CSI data as a 2D heatmap (time × subcarrier).

    Parameters
    ----------
    sample : np.ndarray
        Single CSI sample of shape (250, 90)
    activity_name : str
        Activity label for title
    save_path : Path, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        sample.T,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        origin="lower",
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Subcarrier Index")
    title = "CSI Amplitude Heatmap"
    if activity_name:
        title += f" [{activity_name}]"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="CSI Amplitude")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_spectrogram(
    sample: np.ndarray,
    subcarrier: int = 0,
    activity_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot STFT spectrogram for a given subcarrier.

    NOTE: Uses normalized frequency (fs=1.0) since the UT-HAR
    dataset does not provide a physical sampling frequency.

    Parameters
    ----------
    sample : np.ndarray
        Single CSI sample of shape (250, 90)
    subcarrier : int
        Subcarrier index to visualize
    activity_name : str
        Activity label for title
    save_path : Path, optional
        Path to save the figure
    """
    channel = sample[:, subcarrier]

    freqs, times, Zxx = scipy_stft(
        channel,
        fs=STFT_FS,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
    )

    magnitude = np.abs(Zxx)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(times, freqs, magnitude, shading="gouraud", cmap="magma")
    ax.set_xlabel("Time (sample index)")
    ax.set_ylabel("Normalized Frequency")
    title = f"STFT Spectrogram — Subcarrier {subcarrier}"
    if activity_name:
        title += f" [{activity_name}]"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Magnitude")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot PCA explained variance ratio and cumulative variance.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Explained variance ratio from fitted PCA
    save_path : Path, optional
        Path to save the figure
    """
    cumulative = np.cumsum(explained_variance_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual explained variance
    ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, color="steelblue")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Individual Explained Variance")

    # Cumulative explained variance
    ax2.plot(range(1, len(cumulative) + 1), cumulative, "o-", color="darkorange", markersize=3)
    ax2.axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label="95% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Explained Variance")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    model_name : str
        Model name for title
    save_path : Path, optional
        Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = [ACTIVITY_LABELS[i] for i in range(len(ACTIVITY_LABELS))]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot model comparison bar chart.

    Parameters
    ----------
    metrics : dict
        {model_name: {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}}
    save_path : Path, optional
        Path to save the figure
    """
    model_names = list(metrics.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]

    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)
    colors = sns.color_palette("husl", len(model_names))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (model_name, model_metrics) in enumerate(metrics.items()):
        values = [model_metrics.get(m, 0) for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i], edgecolor="black", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_class_probabilities(
    probabilities: np.ndarray,
    predicted_label: int,
    actual_label: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot class probability distribution for a single prediction.

    Parameters
    ----------
    probabilities : np.ndarray
        Probability for each class (length 7)
    predicted_label : int
        Predicted class label
    actual_label : int, optional
        Ground-truth label
    save_path : Path, optional
        Path to save the figure
    """
    labels = [ACTIVITY_LABELS[i] for i in range(len(probabilities))]
    colors = ["#2ecc71" if i == predicted_label else "#3498db" for i in range(len(probabilities))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, probabilities * 100, color=colors, edgecolor="black", linewidth=0.5)

    # Add percentage labels
    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Probability (%)")
    title = f"Predicted: {ACTIVITY_LABELS[predicted_label]}"
    if actual_label is not None:
        title += f" | Actual: {ACTIVITY_LABELS[actual_label]}"
    ax.set_title(title)
    ax.set_xlim(0, 110)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig
