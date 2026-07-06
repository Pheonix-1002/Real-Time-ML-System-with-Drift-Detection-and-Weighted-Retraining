import streamlit as st
import numpy as np
import time
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy.stats import entropy
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Weighted Retraining ML System", 
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

st.title("🚀 Real-Time ML System with Drift Detection (Weighted Retraining)")
st.markdown("---")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 1rem;
        color: white;
    }
    .warning-text {
        color: #ff6b6b;
        font-weight: bold;
    }
    .success-text {
        color: #51cf66;
        font-weight: bold;
    }
    .info-text {
        color: #4dabf7;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helper Classes
# -----------------------------
class DataStreamSimulator:
    """Simulates real-time data stream with configurable drift"""
    def __init__(self, base_distribution, drift_type="sudden", drift_magnitude=1.0):
        self.base_dist = base_distribution
        self.drift_type = drift_type
        self.drift_magnitude = drift_magnitude
        self.drift_started = False
        self.drift_progress = 0
        
    def generate_batch(self, batch_idx, simulate_drift=False):
        X_new, y_new = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42 + batch_idx
        )
        
        if simulate_drift:
            if self.drift_type == "sudden" and batch_idx > 20:
                X_new += np.random.normal(self.drift_magnitude, 0.5, X_new.shape)
            elif self.drift_type == "gradual":
                if batch_idx > 20:
                    self.drift_progress = min(1.0, self.drift_progress + 0.05)
                    X_new += np.random.normal(self.drift_magnitude * self.drift_progress, 0.5, X_new.shape)
            elif self.drift_type == "recurring":
                if 20 < batch_idx < 30 or 40 < batch_idx < 50:
                    X_new += np.random.normal(self.drift_magnitude, 0.5, X_new.shape)
                    
        return X_new, y_new

class DriftDetector:
    """Multiple drift detection methods"""
    def __init__(self, baseline_dist, threshold=0.5):
        self.baseline_dist = baseline_dist
        self.threshold = threshold
        self.history = deque(maxlen=20)
        
    def detect(self, current_dist):
        # KL Divergence
        kl_score = self._kl_divergence(self.baseline_dist, current_dist)
        
        # Population Stability Index (PSI)
        psi_score = self._calculate_psi(self.baseline_dist, current_dist)
        
        # Moving average of drift scores
        self.history.append(kl_score)
        avg_drift = np.mean(self.history) if len(self.history) > 0 else kl_score
        
        # Weighted combination
        combined_score = 0.7 * kl_score + 0.3 * psi_score
        
        return {
            "kl_divergence": kl_score,
            "psi": psi_score,
            "combined": combined_score,
            "avg_drift": avg_drift,
            "drift_detected": combined_score > self.threshold or avg_drift > self.threshold
        }
    
    @staticmethod
    def _kl_divergence(p, q):
        p = np.abs(p) + 1e-10
        q = np.abs(q) + 1e-10
        p = p / np.sum(p)
        q = q / np.sum(q)
        return entropy(p, q)
    
    @staticmethod
    def _calculate_psi(expected, actual, bins=10):
        """Calculate Population Stability Index"""
        expected = np.abs(expected)
        actual = np.abs(actual)
        
        # Create bins
        hist_expected, bin_edges = np.histogram(expected, bins=bins)
        hist_actual, _ = np.histogram(actual, bins=bin_edges)
        
        # Add small constant to avoid division by zero
        hist_expected = hist_expected + 1e-10
        hist_actual = hist_actual + 1e-10
        
        # Convert to percentages
        hist_expected = hist_expected / np.sum(hist_expected)
        hist_actual = hist_actual / np.sum(hist_actual)
        
        # Calculate PSI
        psi = np.sum((hist_actual - hist_expected) * np.log(hist_actual / hist_expected))
        
        return psi

class ModelManager:
    """Manages model training, evaluation, and retraining"""
    def __init__(self, max_training_samples=10000, weight_decay=0.95):
        self.max_training_samples = max_training_samples
        self.weight_decay = weight_decay
        self.training_history = []
        
    def train_model(self, X, y, weights):
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False
        )
        model.fit(X, y, sample_weight=weights)
        return model
    
    def update_weights(self, current_weights, new_data_weight=2.0):
        # Decay old weights
        updated_weights = current_weights * self.weight_decay
        # Create new weights
        new_weights = np.ones(len(new_data_weight)) * new_data_weight
        
        return np.hstack((updated_weights, new_weights))
    
    def manage_dataset_size(self, X, y, weights):
        """Keep dataset within limit by removing oldest samples"""
        if len(X) > self.max_training_samples:
            excess = len(X) - self.max_training_samples
            X = X[excess:]
            y = y[excess:]
            weights = weights[excess:]
        return X, y, weights

# -----------------------------
# Initialize Data
# -----------------------------
@st.cache_resource
def load_initial_data():
    X, y = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = load_initial_data()

# Training pool
training_X = X_train.copy()
training_y = y_train.copy()
training_weights = np.ones(len(training_y))

# Initialize managers
model_manager = ModelManager()
drift_detector = DriftDetector(np.mean(X_train, axis=0), threshold=0.5)
model = model_manager.train_model(training_X, training_y, training_weights)

# -----------------------------
# Sidebar Configuration
# -----------------------------
st.sidebar.header("⚙ System Configuration")

# Drift settings
st.sidebar.subheader("📊 Drift Configuration")
simulate_drift = st.sidebar.checkbox("Simulate Data Drift", value=True)
drift_type = st.sidebar.selectbox(
    "Drift Type",
    ["sudden", "gradual", "recurring"],
    help="Sudden: abrupt change, Gradual: slow transition, Recurring: periodic drift"
)
drift_magnitude = st.sidebar.slider("Drift Magnitude", 0.5, 3.0, 1.0, 0.5)

# Retraining settings
st.sidebar.subheader("🔄 Retraining Configuration")
retraining_threshold = st.sidebar.slider("Drift Detection Threshold", 0.1, 2.0, 0.5, 0.05)
new_data_weight = st.sidebar.slider("New Data Weight", 1.0, 5.0, 2.0, 0.5)
weight_decay = st.sidebar.slider("Weight Decay Rate", 0.7, 0.99, 0.95, 0.01)

# Streaming settings
st.sidebar.subheader("🎮 Streaming Settings")
n_batches = st.sidebar.slider("Number of Batches", 10, 100, 50, 10)
batch_delay = st.sidebar.slider("Delay between batches (seconds)", 0.1, 2.0, 0.5, 0.1)

# Update parameters
drift_detector.threshold = retraining_threshold
model_manager.weight_decay = weight_decay

run_stream = st.sidebar.button("🚀 Start Streaming", type="primary")
reset_system = st.sidebar.button("🔄 Reset System")

if reset_system:
    st.cache_resource.clear()
    st.rerun()

# -----------------------------
# Main Display Area
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 System Status")
    status_placeholder = st.empty()
    
with col2:
    st.markdown("### 🎯 Current Performance")
    performance_placeholder = st.empty()
    
with col3:
    st.markdown("### 🔍 Drift Detection")
    drift_placeholder = st.empty()

# Create placeholders for charts
chart1_placeholder = st.empty()
chart2_placeholder = st.empty()
chart3_placeholder = st.empty()
metrics_placeholder = st.empty()

# -----------------------------
# Streaming Execution
# -----------------------------
if run_stream:
    
    # Reset tracking variables
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    auc_list = []
    drift_scores = []
    retraining_events = []
    processing_times = []
    dataset_sizes = []
    batch_times = []
    
    # Initialize stream simulator
    stream_simulator = DataStreamSimulator(
        np.mean(X_train, axis=0),
        drift_type=drift_type,
        drift_magnitude=drift_magnitude
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    
    for batch_idx in range(n_batches):
        
        batch_start_time = time.time()
        
        # Generate new batch
        X_new, y_new = stream_simulator.generate_batch(batch_idx, simulate_drift)
        
        # Make predictions
        y_pred = model.predict(X_new)
        y_pred_proba = model.predict_proba(X_new)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate all metrics
        accuracy = accuracy_score(y_new, y_pred)
        precision = precision_score(y_new, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_new, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_new, y_pred, average='weighted', zero_division=0)
        
        # Calculate AUC if binary classification
        auc = roc_auc_score(y_new, y_pred_proba) if y_pred_proba is not None else 0.0
        
        # Store metrics
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc)
        
        # Detect drift
        current_dist = np.mean(X_new, axis=0)
        drift_results = drift_detector.detect(current_dist)
        drift_scores.append(drift_results["combined"])
        
        # Update status display
        status_text.markdown(f"""
        <div style="background-color:#f0f2f6; padding:10px; border-radius:5px;">
        <b>Processing Batch {batch_idx + 1}/{n_batches}</b><br>
        📈 Accuracy: <span class="success-text">{accuracy:.3f}</span> | 
        🎯 Precision: {precision:.3f} | 
        🔍 Recall: {recall:.3f} | 
        ⭐ F1-Score: {f1:.3f}<br>
        ⚡ Drift Score: <span class="{'warning-text' if drift_results['drift_detected'] else 'info-text'}">{drift_results['combined']:.3f}</span> | 
        🚨 Drift Detected: <span class="{'warning-text' if drift_results['drift_detected'] else 'success-text'}">{drift_results['drift_detected']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Weighted retraining logic
        if drift_results["drift_detected"]:
            retraining_events.append(batch_idx)
            
            # Add new data to training pool
            training_X = np.vstack((training_X, X_new))
            training_y = np.hstack((training_y, y_new))
            
            # Update weights
            training_weights = model_manager.update_weights(
                training_weights, 
                np.ones(len(y_new)) * new_data_weight
            )
            
            # Manage dataset size
            training_X, training_y, training_weights = model_manager.manage_dataset_size(
                training_X, training_y, training_weights
            )
            
            # Retrain model
            retrain_start = time.time()
            model = model_manager.train_model(training_X, training_y, training_weights)
            retrain_time = time.time() - retrain_start
            
            processing_times.append(retrain_time)
            
            # Show retraining alert
            drift_placeholder.warning(f"⚠️ Drift detected at batch {batch_idx + 1}! Retraining model... (took {retrain_time:.2f}s)")
        else:
            drift_placeholder.info(f"✅ No drift detected at batch {batch_idx + 1}")
        
        # Record dataset size
        dataset_sizes.append(len(training_X))
        
        # Calculate processing time
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # Update displays every 5 batches or at the end
        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            
            # Create comprehensive dashboard with all metrics
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle('Real-Time ML System Performance Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Accuracy trend
            axes[0, 0].plot(accuracy_list, 'b-', linewidth=2, label='Accuracy')
            window_size = min(5, len(accuracy_list))
            if len(accuracy_list) >= window_size:
                smoothed = np.convolve(accuracy_list, np.ones(window_size)/window_size, mode='valid')
                axes[0, 0].plot(range(window_size-1, len(accuracy_list)), smoothed, 'g--', linewidth=2, label=f'Smoothed (w={window_size})')
            axes[0, 0].axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Target (0.8)')
            for event in retraining_events:
                if event < len(accuracy_list):
                    axes[0, 0].axvline(x=event, color='orange', alpha=0.5, linestyle='--', linewidth=1)
            axes[0, 0].set_xlabel("Batch Number")
            axes[0, 0].set_ylabel("Score")
            axes[0, 0].set_title("Model Accuracy Over Time")
            axes[0, 0].legend(loc='lower right')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim([0, 1])
            
            # 2. Precision, Recall, F1 trends
            axes[0, 1].plot(precision_list, 'g-', linewidth=2, label='Precision')
            axes[0, 1].plot(recall_list, 'b-', linewidth=2, label='Recall')
            axes[0, 1].plot(f1_list, 'm-', linewidth=2, label='F1-Score')
            axes[0, 1].set_xlabel("Batch Number")
            axes[0, 1].set_ylabel("Score")
            axes[0, 1].set_title("Precision, Recall & F1-Score")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1])
            
            # 3. Drift detection
            axes[0, 2].plot(drift_scores, 'r-', linewidth=2, label='Drift Score')
            axes[0, 2].axhline(y=retraining_threshold, color='orange', linestyle='--', 
                              label=f'Threshold ({retraining_threshold})')
            axes[0, 2].fill_between(range(len(drift_scores)), 0, drift_scores, 
                                   where=np.array(drift_scores) > retraining_threshold, 
                                   color='red', alpha=0.3, label='Drift Region')
            axes[0, 2].set_xlabel("Batch Number")
            axes[0, 2].set_ylabel("Drift Score")
            axes[0, 2].set_title("Concept Drift Detection")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. ROC-AUC (if available)
            axes[1, 0].plot(auc_list, 'c-', linewidth=2, label='ROC-AUC')
            axes[1, 0].axhline(y=0.85, color='r', linestyle='--', alpha=0.7, label='Good Threshold (0.85)')
            axes[1, 0].set_xlabel("Batch Number")
            axes[1, 0].set_ylabel("AUC Score")
            axes[1, 0].set_title("ROC-AUC Score")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 1])
            
            # 5. Processing time and dataset size
            ax5_1 = axes[1, 1]
            ax5_2 = ax5_1.twinx()
            ax5_1.plot(batch_times, 'purple', linewidth=2, label='Batch Time')
            ax5_1.set_xlabel("Batch Number")
            ax5_1.set_ylabel("Time (seconds)", color='purple')
            ax5_1.tick_params(axis='y', labelcolor='purple')
            ax5_2.plot(dataset_sizes, 'brown', linewidth=2, linestyle='--', label='Dataset Size')
            ax5_2.set_ylabel("Dataset Size", color='brown')
            ax5_2.tick_params(axis='y', labelcolor='brown')
            axes[1, 1].set_title("Processing Metrics")
            ax5_1.grid(True, alpha=0.3)
            
            # 6. Confusion Matrix for latest batch
            cm = confusion_matrix(y_new, y_pred)
            im = axes[1, 2].imshow(cm, interpolation='nearest', cmap='Blues')
            axes[1, 2].set_title(f'Confusion Matrix (Batch {batch_idx + 1})')
            axes[1, 2].set_xlabel('Predicted')
            axes[1, 2].set_ylabel('Actual')
            
            # Add text annotations to confusion matrix
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[1, 2].text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
            
            plt.tight_layout()
            chart1_placeholder.pyplot(fig)
            plt.close()
            
            # Additional metrics summary
            with metrics_placeholder.container():
                st.markdown("### 📊 Current Performance Metrics")
                col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
                
                with col_a:
                    st.metric("Accuracy", f"{accuracy:.3f}", 
                             delta=f"{accuracy - accuracy_list[-2] if len(accuracy_list) > 1 else 0:.3f}")
                with col_b:
                    st.metric("Precision", f"{precision:.3f}")
                with col_c:
                    st.metric("Recall", f"{recall:.3f}")
                with col_d:
                    st.metric("F1-Score", f"{f1:.3f}")
                with col_e:
                    st.metric("ROC-AUC", f"{auc:.3f}")
                with col_f:
                    st.metric("Drift Score", f"{drift_results['combined']:.3f}", 
                             delta="Drift!" if drift_results['drift_detected'] else "Stable")
        
        # Update progress
        progress_bar.progress((batch_idx + 1) / n_batches)
        time.sleep(batch_delay)
    
    # Final summary
    total_time = time.time() - start_time
    
    st.markdown("---")
    st.success("✅ Streaming Completed!")
    
    # Display final statistics
    st.markdown("## 📈 Final Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Accuracy", f"{accuracy_list[-1]:.3f}", 
                 delta=f"{np.mean(accuracy_list[-10:]):.3f}" if len(accuracy_list) >= 10 else "N/A")
    with col2:
        st.metric("Average Accuracy", f"{np.mean(accuracy_list):.3f}")
    with col3:
        st.metric("Final F1-Score", f"{f1_list[-1]:.3f}")
    with col4:
        st.metric("Total Retraining Events", len(retraining_events))
    
    # Performance summary dataframe
    summary_data = {
        "Metric": ["Final Accuracy", "Average Accuracy", "Accuracy Std Dev", 
                   "Final Precision", "Average Precision",
                   "Final Recall", "Average Recall",
                   "Final F1-Score", "Average F1-Score",
                   "Final ROC-AUC", "Average ROC-AUC",
                   "Retraining Rate", "Avg Processing Time", "Total Batches"],
        "Value": [
            f"{accuracy_list[-1]:.3f}",
            f"{np.mean(accuracy_list):.3f}",
            f"{np.std(accuracy_list):.3f}",
            f"{precision_list[-1]:.3f}",
            f"{np.mean(precision_list):.3f}",
            f"{recall_list[-1]:.3f}",
            f"{np.mean(recall_list):.3f}",
            f"{f1_list[-1]:.3f}",
            f"{np.mean(f1_list):.3f}",
            f"{auc_list[-1]:.3f}",
            f"{np.mean(auc_list):.3f}",
            f"{(len(retraining_events)/n_batches)*100:.1f}%",
            f"{np.mean(processing_times) if processing_times else 0:.3f}s",
            n_batches
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Performance comparison chart
    st.markdown("### 📊 Performance Comparison")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    final_values = [accuracy_list[-1], precision_list[-1], recall_list[-1], f1_list[-1], auc_list[-1]]
    avg_values = [np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_list), np.mean(auc_list)]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, final_values, width, label='Final Batch', color='#2ecc71')
    bars2 = ax2.bar(x + width/2, avg_values, width, label='Average', color='#3498db')
    
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Score')
    ax2.set_title('Model Performance: Final vs Average')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    st.pyplot(fig2)
    plt.close()
    
    # Download results button
    results = {
        "accuracies": accuracy_list,
        "precisions": precision_list,
        "recalls": recall_list,
        "f1_scores": f1_list,
        "auc_scores": auc_list,
        "drift_scores": drift_scores,
        "retraining_events": retraining_events,
        "processing_times": processing_times,
        "dataset_sizes": dataset_sizes,
        "batch_times": batch_times,
        "config": {
            "drift_type": drift_type,
            "drift_magnitude": drift_magnitude,
            "threshold": retraining_threshold,
            "n_batches": n_batches
        }
    }
    
    st.download_button(
        label="📥 Download Complete Results (JSON)",
        data=str(results),
        file_name=f"ml_system_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

else:
    st.info("👈 Configure settings in the sidebar and click 'Start Streaming' to begin")
    
    # Show system overview
    with st.expander("📖 System Overview", expanded=True):
        st.markdown("""
        ### 🎯 Real-Time ML System with Drift Detection and Weighted Retraining
        
        This system demonstrates an adaptive machine learning pipeline that detects and responds to concept drift:
        
        #### 🔄 How It Works
        
        1. **Data Streaming**: Simulates real-time data batches (200 samples each)
        2. **Drift Detection**: Uses KL Divergence and PSI metrics to detect distribution changes
        3. **Weighted Retraining**: When drift is detected, the model retrains with:
           - Decayed weights for older samples (reduces influence)
           - Higher weights for new samples (prioritizes recent data)
           - Controlled dataset size management (prevents memory bloat)
        
        #### 📊 Metrics Tracked
        
        - **Accuracy**: Overall classification accuracy
        - **Precision**: Positive predictive value
        - **Recall**: True positive rate (sensitivity)
        - **F1-Score**: Harmonic mean of precision and recall
        - **ROC-AUC**: Area under ROC curve
        - **Drift Score**: KL divergence + PSI combined score
        
        #### 🎮 Available Drift Types
        
        - **Sudden**: Abrupt distribution change after batch 20
        - **Gradual**: Slow transition over multiple batches
        - **Recurring**: Periodic drift that appears and disappears
        
        #### 📈 Visualizations
        
        - Real-time performance trends (all metrics)
        - Drift detection monitoring
        - Confusion matrix for latest batch
        - Processing time and dataset size tracking
        - Final performance comparison charts
        
        #### 🎯 Use Cases
        
        - Financial fraud detection (evolving patterns)
        - User behavior prediction (changing preferences)
        - Sensor data analysis (equipment degradation)
        - Recommendation systems (trending items)
        - Anomaly detection in streaming data
        
        #### 🚀 Getting Started
        
        1. Adjust configuration parameters in the sidebar
        2. Click "Start Streaming" to begin
        3. Watch real-time updates as data streams in
        4. Observe drift detection and automatic retraining
        5. Download results for analysis
        """)
    
    # Show sample configuration
    with st.expander("⚙️ Configuration Guide"):
        st.markdown("""
        ### Recommended Settings for Different Scenarios
        
        | Scenario | Drift Type | Threshold | Weight Decay | New Data Weight |
        |----------|-----------|-----------|--------------|-----------------|
        | Sensitive Detection | Sudden | 0.3 | 0.95 | 2.5 |
        | Conservative | Gradual | 0.7 | 0.90 | 1.5 |
        | Rapid Adaptation | Sudden | 0.4 | 0.85 | 3.0 |
        | Stable Environment | None | 0.8 | 0.98 | 1.0 |
        | Recurring Patterns | Recurring | 0.5 | 0.92 | 2.0 |
        
        ### Understanding Parameters
        
        - **Drift Threshold**: Lower = more sensitive to changes (more retraining)
        - **Weight Decay**: Lower = older data forgotten faster
        - **New Data Weight**: Higher = new samples have more influence
        - **Batch Delay**: Controls simulation speed (seconds between batches)
        """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 10px;">
    🔧 <b>Advanced ML System</b> | Built with Streamlit & XGBoost | 
    Real-time Drift Detection & Weighted Retraining |
    📊 Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
</div>
""", unsafe_allow_html=True)