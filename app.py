"""
WiFi CSI Human Activity Recognition — Streamlit Application

A demonstration application for the mid-semester evaluation of the
B.Tech major project: "Human and Object Recognition Using WiFi CSI Signals"

This app loads PRE-TRAINED models and provides:
1. Dataset overview
2. CSI visualization
3. Single sample prediction with class probabilities
4. Model performance metrics
5. Confusion matrix display

IMPORTANT: This app does NOT retrain models. All models, scalers,
and PCA objects are loaded from saved files.
"""

import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import ACTIVITY_LABELS, NUM_CLASSES, NUM_TIMESTEPS, NUM_SUBCARRIERS
from src.predict import load_pipeline, predict_sample
from src.feature_extraction import get_feature_info
from scipy.signal import stft as scipy_stft
from src.feature_extraction import STFT_FS, STFT_NPERSEG, STFT_NOVERLAP


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="WiFi CSI Human Activity Recognition",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .correct-prediction {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .wrong-prediction {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Cached data loading functions
# ============================================================
@st.cache_data
def load_test_data():
    """Load test data (cached to avoid reloading)."""
    data_dir = PROJECT_ROOT / "data"
    X_test = np.load(str(data_dir / "X_test.npy"), allow_pickle=True)
    y_test = np.load(str(data_dir / "y_test.npy"), allow_pickle=True)
    return X_test, y_test


@st.cache_data
def load_train_labels():
    """Load training labels for dataset overview."""
    data_dir = PROJECT_ROOT / "data"
    y_train = np.load(str(data_dir / "y_train.npy"), allow_pickle=True)
    return y_train


@st.cache_resource
def load_cached_pipeline():
    """Load the prediction pipeline (cached to avoid reloading)."""
    return load_pipeline(PROJECT_ROOT / "models")


@st.cache_data
def load_metrics():
    """Load saved model metrics."""
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


# ============================================================
# Main Application
# ============================================================
def main():
    # Header
    st.markdown('<div class="main-header">📡 WiFi CSI Human Activity Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">B.Tech Major Project — Mid-Semester Evaluation<br>'
        'Using UT-HAR Dataset with Classical Machine Learning</div>',
        unsafe_allow_html=True,
    )

    # Load data and pipeline
    try:
        X_test, y_test = load_test_data()
        y_train = load_train_labels()
        pipeline = load_cached_pipeline()
        metrics = load_metrics()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.info("Please run the training pipeline first: `python -m src.train`")
        return

    # ============================================================
    # Sidebar
    # ============================================================
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("📊 Dataset Information")
        st.write(f"**Training samples:** {len(y_train)}")
        st.write(f"**Test samples:** {len(X_test)}")
        st.write(f"**Classes:** {NUM_CLASSES}")
        st.write(f"**Timesteps:** {NUM_TIMESTEPS}")
        st.write(f"**Subcarriers:** {NUM_SUBCARRIERS}")

        st.divider()

        st.subheader("🤖 Model Selection")
        available_models = list(pipeline["models"].keys())
        selected_model = st.selectbox("Select Model", available_models)

        st.divider()

        st.subheader("🔬 Test Sample")
        sample_idx = st.number_input(
            "Test Sample Index",
            min_value=0,
            max_value=len(X_test) - 1,
            value=25,
            step=1,
            help=f"Select a test sample (0 to {len(X_test) - 1})",
        )

        predict_button = st.button("🎯 Predict Activity", type="primary", use_container_width=True)

        st.divider()

        st.subheader("📋 Activity Classes")
        for label, name in ACTIVITY_LABELS.items():
            st.write(f"**{label}:** {name}")

        st.divider()

        st.subheader("🔧 Mid-Sem Scope")
        st.caption(
            "**Current:** UT-HAR, Statistical/DWT/STFT features, PCA, "
            "Random Forest, SVM, KNN"
        )
        st.caption(
            "**Future:** CNN, LSTM, CNN-LSTM, Object Recognition, "
            "Real-time CSI acquisition"
        )

    # ============================================================
    # Main Content — Tabs
    # ============================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Prediction",
        "📊 Dataset Overview",
        "📈 Model Performance",
        "🔬 Feature Info",
    ])

    # ============================================================
    # Tab 1: Prediction
    # ============================================================
    with tab1:
        if predict_button:
            sample = X_test[sample_idx]
            actual_label = int(y_test[sample_idx])
            actual_activity = ACTIVITY_LABELS[actual_label]

            # Make prediction
            with st.spinner("Extracting features and predicting..."):
                result = predict_sample(sample, pipeline, model_name=selected_model)

            predicted_activity = result["predicted_activity"]
            confidence = result["confidence"]
            is_correct = result["predicted_label"] == actual_label

            # Prediction result box
            st.subheader("Prediction Result")
            col1, col2, col3 = st.columns(3)

            with col1:
                css_class = "correct-prediction" if is_correct else "wrong-prediction"
                icon = "✅" if is_correct else "❌"
                st.markdown(
                    f'<div class="{css_class}">'
                    f'<h3>{icon} Predicted Activity</h3>'
                    f'<h2>{predicted_activity}</h2>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<h3>🎯 Actual Activity</h3>'
                    f'<h2>{actual_activity}</h2>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<h3>📊 Confidence</h3>'
                    f'<h2>{confidence * 100:.1f}%</h2>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.divider()

            # Class probabilities chart
            col_prob, col_csi = st.columns([1, 1])

            with col_prob:
                st.subheader("Class Probabilities")
                probs = result["probabilities"]
                activity_names = [ACTIVITY_LABELS[i] for i in range(len(probs))]

                # Create horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=probs * 100,
                    y=activity_names,
                    orientation="h",
                    marker_color=[
                        "#2ecc71" if i == result["predicted_label"] else "#3498db"
                        for i in range(len(probs))
                    ],
                    text=[f"{p*100:.1f}%" for p in probs],
                    textposition="outside",
                ))
                fig.update_layout(
                    xaxis_title="Probability (%)",
                    yaxis_title="",
                    height=350,
                    margin=dict(l=0, r=50, t=10, b=30),
                    xaxis=dict(range=[0, max(probs * 100) + 15]),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_csi:
                st.subheader("CSI Heatmap")
                fig_heat, ax_heat = plt.subplots(figsize=(8, 4))
                im = ax_heat.imshow(
                    sample.T, aspect="auto", cmap="viridis",
                    interpolation="nearest", origin="lower",
                )
                ax_heat.set_xlabel("Time Step")
                ax_heat.set_ylabel("Subcarrier Index")
                ax_heat.set_title(f"CSI Amplitude — Sample {sample_idx} [{actual_activity}]")
                plt.colorbar(im, ax=ax_heat, label="Amplitude")
                plt.tight_layout()
                st.pyplot(fig_heat)
                plt.close(fig_heat)

            st.divider()

            # CSI Waveform and Spectrogram
            col_wave, col_spec = st.columns([1, 1])

            with col_wave:
                st.subheader("CSI Waveform (Selected Subcarriers)")
                subcarriers = [0, 15, 30, 45, 60, 75]
                fig_wave = go.Figure()
                for sc in subcarriers:
                    fig_wave.add_trace(go.Scatter(
                        y=sample[:, sc],
                        mode="lines",
                        name=f"SC {sc}",
                        line=dict(width=1),
                    ))
                fig_wave.update_layout(
                    xaxis_title="Time Step",
                    yaxis_title="CSI Amplitude",
                    height=400,
                    margin=dict(l=0, r=0, t=10, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_wave, use_container_width=True)

            with col_spec:
                st.subheader("STFT Spectrogram (Subcarrier 0)")
                channel = sample[:, 0]
                freqs, times, Zxx = scipy_stft(
                    channel, fs=STFT_FS, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
                )
                magnitude = np.abs(Zxx)

                fig_spec, ax_spec = plt.subplots(figsize=(8, 4))
                ax_spec.pcolormesh(times, freqs, magnitude, shading="gouraud", cmap="magma")
                ax_spec.set_xlabel("Time (sample index)")
                ax_spec.set_ylabel("Normalized Frequency")
                ax_spec.set_title(f"STFT Spectrogram — [{actual_activity}]")
                plt.tight_layout()
                st.pyplot(fig_spec)
                plt.close(fig_spec)

        else:
            st.info(
                "👈 Select a test sample index in the sidebar and click "
                "**Predict Activity** to see results."
            )

    # ============================================================
    # Tab 2: Dataset Overview
    # ============================================================
    with tab2:
        st.subheader("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training Samples", len(y_train))
        col2.metric("Test Samples", len(X_test))
        col3.metric("Classes", NUM_CLASSES)
        col4.metric("Feature Shape", f"{NUM_TIMESTEPS} × {NUM_SUBCARRIERS}")

        st.divider()

        # Class distribution
        col_train, col_test = st.columns(2)

        with col_train:
            st.subheader("Training Set Distribution")
            unique, counts = np.unique(y_train, return_counts=True)
            labels = [ACTIVITY_LABELS[int(u)] for u in unique]

            fig_td = px.bar(
                x=labels, y=counts,
                color=labels,
                labels={"x": "Activity", "y": "Samples"},
                title="Training Set — Class Distribution",
            )
            fig_td.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_td, use_container_width=True)

        with col_test:
            st.subheader("Test Set Distribution")
            unique, counts = np.unique(y_test, return_counts=True)
            labels = [ACTIVITY_LABELS[int(u)] for u in unique]

            fig_ts = px.bar(
                x=labels, y=counts,
                color=labels,
                labels={"x": "Activity", "y": "Samples"},
                title="Test Set — Class Distribution",
            )
            fig_ts.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_ts, use_container_width=True)

        st.divider()

        # Sample CSI visualization for each activity
        st.subheader("Sample CSI Patterns by Activity")
        selected_activity = st.selectbox(
            "Select Activity",
            [ACTIVITY_LABELS[i] for i in range(NUM_CLASSES)],
        )
        selected_label = [k for k, v in ACTIVITY_LABELS.items() if v == selected_activity][0]

        # Find a sample of this activity in test set
        matching_indices = np.where(y_test == selected_label)[0]
        if len(matching_indices) > 0:
            demo_idx = matching_indices[0]
            demo_sample = X_test[demo_idx]

            col_h, col_w = st.columns(2)
            with col_h:
                fig_d, ax_d = plt.subplots(figsize=(8, 4))
                ax_d.imshow(
                    demo_sample.T, aspect="auto", cmap="viridis",
                    interpolation="nearest", origin="lower",
                )
                ax_d.set_xlabel("Time Step")
                ax_d.set_ylabel("Subcarrier Index")
                ax_d.set_title(f"CSI Heatmap — {selected_activity}")
                plt.tight_layout()
                st.pyplot(fig_d)
                plt.close(fig_d)

            with col_w:
                fig_w = go.Figure()
                for sc in [0, 30, 60, 89]:
                    fig_w.add_trace(go.Scatter(
                        y=demo_sample[:, sc], mode="lines",
                        name=f"SC {sc}", line=dict(width=1),
                    ))
                fig_w.update_layout(
                    xaxis_title="Time Step", yaxis_title="CSI Amplitude",
                    title=f"CSI Waveform — {selected_activity}",
                    height=400,
                )
                st.plotly_chart(fig_w, use_container_width=True)

    # ============================================================
    # Tab 3: Model Performance
    # ============================================================
    with tab3:
        if metrics:
            st.subheader("Model Performance Comparison")

            # Metrics table
            metric_names = ["accuracy", "precision", "recall", "f1"]
            cols = st.columns(len(metrics))

            for i, (model_name, model_metrics) in enumerate(metrics.items()):
                with cols[i]:
                    st.markdown(f"### {model_name}")
                    for mn in metric_names:
                        val = model_metrics.get(mn, 0)
                        st.metric(mn.capitalize(), f"{val:.4f}")

            st.divider()

            # Comparison chart
            st.subheader("Performance Comparison Chart")
            model_names = list(metrics.keys())
            fig_comp = go.Figure()

            colors = px.colors.qualitative.Set2
            for i, mn in enumerate(metric_names):
                values = [metrics[m].get(mn, 0) for m in model_names]
                fig_comp.add_trace(go.Bar(
                    name=mn.capitalize(),
                    x=model_names,
                    y=values,
                    text=[f"{v:.3f}" for v in values],
                    textposition="outside",
                    marker_color=colors[i],
                ))

            fig_comp.update_layout(
                barmode="group",
                yaxis=dict(range=[0, 1.15], title="Score"),
                xaxis_title="Model",
                height=500,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            st.divider()

            # Confusion matrices from saved figures
            st.subheader("Confusion Matrices")
            fig_dir = PROJECT_ROOT / "results" / "figures"
            cm_cols = st.columns(len(metrics))
            for i, model_name in enumerate(metrics.keys()):
                with cm_cols[i]:
                    cm_path = fig_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
                    if cm_path.exists():
                        st.image(str(cm_path), caption=model_name)
                    else:
                        st.warning(f"Confusion matrix not found for {model_name}")

        else:
            st.warning("No metrics found. Please run the training pipeline first.")

    # ============================================================
    # Tab 4: Feature Info
    # ============================================================
    with tab4:
        st.subheader("Feature Extraction Configuration")

        feat_info = get_feature_info()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📊 Statistical Features")
            info = feat_info["statistical"]
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Subcarriers used:** {info['num_subcarriers']}")
            st.write(f"**Features per subcarrier:** {info['features_per_subcarrier']}")
            st.write(f"**Total features:** {info['total']}")
            st.write("**Features:**")
            for fn in info["feature_names"]:
                st.write(f"  • {fn}")

        with col2:
            st.markdown("### 🌊 DWT Features")
            info = feat_info["dwt"]
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Wavelet:** {info['wavelet']}")
            st.write(f"**Decomposition level:** {info['level']}")
            st.write(f"**Subcarriers selected:** {info['selected_subcarriers']} (every {info['subcarrier_step']}th)")
            st.write(f"**Features per subcarrier:** {info['features_per_subcarrier']}")
            st.write(f"**Total features:** {info['total']}")

        with col3:
            st.markdown("### 📈 STFT Features")
            info = feat_info["stft"]
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Window size:** {info['nperseg']}")
            st.write(f"**Overlap:** {info['noverlap']}")
            st.write(f"**Sampling frequency:** {info['fs']}")
            st.write(f"**Subcarriers selected:** {info['selected_subcarriers']}")
            st.write(f"**Features per subcarrier:** {info['features_per_subcarrier']}")
            st.write(f"**Total features:** {info['total']}")

        st.divider()

        # PCA info
        st.subheader("PCA Dimensionality Reduction")
        pca_fig_path = PROJECT_ROOT / "results" / "figures" / "pca_explained_variance.png"
        if pca_fig_path.exists():
            st.image(str(pca_fig_path), caption="PCA Explained Variance")
        else:
            st.info("PCA variance plot not available. Run training first.")

        st.markdown("""
        **Data Leakage Prevention:**
        - StandardScaler: fit on training data **only**, used to transform test data
        - PCA: fit on training data **only**, used to transform test data
        - Models: trained on training data **only**, evaluated on test data
        """)


if __name__ == "__main__":
    main()
