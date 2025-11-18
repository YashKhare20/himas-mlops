"""
HIMAS Federated Model Evaluation System
========================================

Evaluates trained federated learning model on held-out test data.

This module provides comprehensive model evaluation with:
- Configurable prediction thresholds via command-line or configuration
- Hospital-specific and aggregated performance metrics
- ROC curves, confusion matrices, and performance comparisons
- Protection against data leakage through correct preprocessing

Key Features:
-------------
1. Data Leakage Prevention:
   - Fits preprocessor on combined training data from all hospitals
   - Transforms test data using training statistics only
   - Maintains scientific validity of performance estimates

2. Configurable Thresholds:
   - Default threshold from pyproject.toml
   - Override via command-line argument
   - Enables testing different thresholds for clinical scenarios

3. Comprehensive Metrics:
   - ROC AUC for discrimination assessment
   - Sensitivity (recall) and specificity for clinical safety
   - PPV (precision) and NPV for prediction confidence
   - Confusion matrices showing all classification outcomes

4. Multi-Format Output:
   - JSON for programmatic access and archival
   - PNG visualizations for presentations
   - Detailed logging for audit trails

Usage:
------
    # Evaluate with default threshold
    python evaluate_model.py

    # Evaluate with custom threshold
    python evaluate_model.py --threshold 0.45

    # Evaluate with optimized threshold (after running optimize_threshold.py)
    python evaluate_model.py --threshold 0.467

Author: HIMAS Team
License: Apache 2.0
"""

import argparse
import logging
import numpy as np
import pandas as pd
import toml
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import keras
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

import os
import mlflow
from mlflow.tracking import MlflowClient  # <-- NEW

_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
_MLFLOW_EXP = os.getenv("MLFLOW_EXPERIMENT_NAME", "himas-federated-eval")
mlflow.set_tracking_uri(_MLFLOW_URI)
mlflow.set_experiment(_MLFLOW_EXP)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config() -> Dict:
    """Load configuration from pyproject.toml."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        pyproject_path = parent / 'pyproject.toml'
        if pyproject_path.exists():
            with open(pyproject_path, 'r') as f:
                return toml.load(f)
    raise FileNotFoundError("pyproject.toml not found")


def get_config_value(key_path: str, default=None):
    """Get configuration value using dot notation path."""
    config = load_config()
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

def get_latest_model_path() -> Path:
    """
    Get path to latest model based on timestamp.

    Args:
        hospital_name: Hospital name to search in hyper-{hospital_name} folder.
                      If None, uses first hospital from config.

    Returns:
        Path to latest .keras model file

    Raises:
        FileNotFoundError: If no model files found
    """

    # Get model directory from config
    models_base_dir = Path(get_config_value(
        'tool.himas.paths.models-dir', 'models'))
    eval_model_dir = get_config_value(
        'tool.himas.paths.eval-model-dir', 'hyper-hospital_a')

    model_dir = models_base_dir / eval_model_dir

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find all .keras files
    model_files = list(model_dir.glob('*.keras'))

    if not model_files:
        raise FileNotFoundError(f"No .keras model files found in {model_dir}")

    # Sort by modification time (latest first)
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

    logger.info(f"Found latest model: {latest_model.name}")
    return latest_model


def load_hyperparameters_metadata() -> Optional[Dict]:
    """
    Load hyperparameters used for model training.

    Returns:
        Dictionary of hyperparameters or None if file not found

    Note:
        Hyperparameters are stored separately from model weights for:
        - Reproducibility documentation
        - Model architecture verification
        - Audit trail for regulatory compliance
    """
    hp_path = get_config_value('tool.flwr.app.config.shared-hyperparameters')
    if not hp_path or not Path(hp_path).exists():
        return None
    with open(hp_path, 'r') as f:
        return json.load(f)


# Load configuration constants
PROJECT_ID = get_config_value('tool.himas.data.project-id')
DATASET_ID = get_config_value('tool.himas.data.dataset-id', 'federated')
HOSPITALS = get_config_value(
    'tool.himas.data.hospital-names', ['hospital_a', 'hospital_b', 'hospital_c'])
NUMERICAL_FEATURES = get_config_value('tool.himas.data.numerical-features')
CATEGORICAL_FEATURES = get_config_value('tool.himas.data.categorical-features')
TARGET = get_config_value(
    'tool.himas.data.target-column', 'icu_mortality_label')


class DataPreprocessor:
    """
    Data preprocessor implementing scikit-learn best practices.

    CRITICAL FOR PREVENTING DATA LEAKAGE:
    - fit() learns statistics from training data ONLY
    - transform() applies learned statistics to test data
    - NEVER call fit() or fit_transform() on test data

    This pattern ensures test data characteristics don't influence preprocessing,
    maintaining valid performance estimates for production deployment.
    """

    def __init__(self):
        """Initialize unfitted preprocessor."""
        self.numerical_scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Fit preprocessing on training data exclusively.

        Learns:
            - Numerical features: mean and standard deviation for z-score normalization
            - Categorical features: unique categories and integer encodings

        Args:
            df_train: Training dataframe from BigQuery

        Warning:
            NEVER call this method on validation or test data.
            Doing so causes data leakage and invalidates performance metrics.
        """
        logger.info(
            f"Fitting preprocessor on {len(df_train):,} TRAINING samples")

        excluded = get_config_value('tool.himas.data.excluded-columns', [])
        exclude_cols = [TARGET] + excluded
        X_train = df_train.drop(
            columns=[col for col in exclude_cols if col in df_train.columns])

        # Fit numerical scaler
        X_numerical = X_train[NUMERICAL_FEATURES].copy()
        for col in NUMERICAL_FEATURES:
            if col in X_numerical.columns:
                X_numerical[col] = pd.to_numeric(
                    X_numerical[col], errors='coerce').astype('float64')
        X_numerical = X_numerical.fillna(X_numerical.median())
        self.numerical_scaler.fit(X_numerical)  # Learns mean and std

        # Fit label encoders
        for col in CATEGORICAL_FEATURES:
            if col in X_train.columns:
                self.label_encoders[col] = LabelEncoder()
                col_data = X_train[col].fillna('Unknown').astype(str)
                # Learns category mappings
                self.label_encoders[col].fit(col_data)

        self.is_fitted = True
        logger.info(
            "Preprocessor fitted (statistics learned from training data only)")

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted preprocessing (correct for test data).

        Applies transformations learned from training data without fitting.
        This is the CORRECT approach for test data preprocessing.

        Args:
            df: Test or validation dataframe

        Returns:
            Tuple of (X_processed, y)

        Raises:
            ValueError: If called before fit()
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        excluded = get_config_value('tool.himas.data.excluded-columns', [])
        exclude_cols = [TARGET] + excluded
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[TARGET].values

        # Transform numerical (uses TRAINING mean/std)
        X_numerical = X[NUMERICAL_FEATURES].copy()
        for col in NUMERICAL_FEATURES:
            if col in X_numerical.columns:
                X_numerical[col] = pd.to_numeric(
                    X_numerical[col], errors='coerce').astype('float64')
        X_numerical = X_numerical.fillna(X_numerical.median())
        X_numerical_scaled = self.numerical_scaler.transform(
            X_numerical)  # Transform only

        # Transform categorical (uses TRAINING encodings)
        X_categorical_encoded = []
        for col in CATEGORICAL_FEATURES:
            if col in X.columns and col in self.label_encoders:
                col_data = X[col].fillna('Unknown').astype(str)
                le = self.label_encoders[col]
                encoded = np.array([
                    le.transform([val])[0] if val in le.classes_ else -1
                    for val in col_data
                ])
                X_categorical_encoded.append(encoded.reshape(-1, 1))

        if X_categorical_encoded:
            X_categorical_array = np.hstack(X_categorical_encoded)
            X_processed = np.hstack([X_numerical_scaled, X_categorical_array])
        else:
            X_processed = X_numerical_scaled

        return X_processed, y


class ModelEvaluator:
    """
    Comprehensive model evaluation system for federated ICU mortality prediction.

    Evaluates model performance on held-out test data with:
    - Configurable prediction thresholds
    - Hospital-specific and aggregated metrics
    - ROC curves and confusion matrices
    - Protection against data leakage

    The evaluation process:
    1. Load trained model and hyperparameters
    2. Fit preprocessor on training data (prevents leakage)
    3. Transform test data using fitted preprocessor
    4. Generate predictions at specified threshold
    5. Compute comprehensive metrics
    6. Create visualizations
    7. Save results in multiple formats

    Attributes:
        model_path: Path to trained Keras model file
        project_id: GCP project for BigQuery access
        threshold: Prediction threshold for binary classification
        client: BigQuery client instance
        model: Loaded Keras model
        preprocessor: Fitted DataPreprocessor for consistent transformation
        results: Dict storing evaluation results per hospital
        hyperparameters: Model architecture and training hyperparameters
        output_dir: Directory for saving results and figures
    """

    def __init__(self, model_path: str, project_id: str, threshold: float = 0.5):
        """
        Initialize model evaluator with configurable threshold.

        Args:
            model_path: Path to saved Keras model (.keras file)
            project_id: Google Cloud Platform project ID
            threshold: Prediction threshold for binary classification (default: 0.5)
                      Lower threshold → higher recall (catch more deaths, more false alarms)
                      Higher threshold → higher precision (fewer false alarms, miss more deaths)
        """
        self.model_path = Path(model_path)
        self.project_id = project_id
        self.threshold = threshold
        self.client = bigquery.Client(project=project_id)
        self.model = None
        self.preprocessor = None
        self.results = {}
        self.hyperparameters = None

        # Create output directory structure
        self.output_dir = Path(get_config_value(
            'tool.himas.paths.evaluation-dir', 'evaluation_results'))
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)

        logger.info(f"ModelEvaluator initialized:")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Prediction threshold: {threshold}")

    def load_model_and_config(self) -> None:
        """
        Load trained model and associated hyperparameters.

        Loads:
            - Keras model from disk
            - Hyperparameters JSON for architecture verification

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        logger.info("="*70)
        logger.info("Loading Model and Configuration")
        logger.info("="*70)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load Keras model
        self.model = keras.models.load_model(str(self.model_path))
        logger.info(f"Model loaded")
        logger.info(f"  Total parameters: {self.model.count_params():,}")

        # Load hyperparameters metadata
        self.hyperparameters = load_hyperparameters_metadata()
        if self.hyperparameters:
            logger.info(f"Hyperparameters loaded")
            logger.info(
                f"  Architecture: {self.hyperparameters.get('architecture')}")
            logger.info(f"  Layers: {self.hyperparameters.get('num_layers')}")
        else:
            logger.warning("Hyperparameters metadata not available")

        logger.info("="*70)

    def fit_preprocessor_on_training_data(self) -> None:
        """
        Fit preprocessor on combined training data to prevent data leakage.

        CRITICAL STEP FOR VALID EVALUATION:
        ------------------------------------
        This method fits the preprocessor (StandardScaler and LabelEncoders)
        exclusively on training data before any test data is accessed.

        Why This Matters:
        -----------------
        If we fit the preprocessor on test data:
        - Test data mean/std would influence z-score normalization
        - Test data categories would influence label encoding
        - Performance metrics would be overly optimistic (data leakage)
        - Results would not generalize to true production deployment

        Implementation:
        ---------------
        1. Load training data from all three hospitals
        2. Combine into single dataset (mirrors federated training)
        3. Fit preprocessor on combined training data
        4. Use fitted preprocessor to transform test data later

        Note:
            Combining training data for evaluation preprocessing is acceptable
            because we're not training on it - just learning transformation parameters.
            This differs from federated TRAINING where each hospital maintains
            separate preprocessors for data privacy.
        """
        logger.info("="*70)
        logger.info(
            "Fitting Preprocessor on Training Data (Preventing Data Leakage)")
        logger.info("="*70)

        self.preprocessor = DataPreprocessor()

        # Load training data from all hospitals
        train_split = get_config_value('tool.himas.data.train-split', 'train')
        all_train_data = []

        for hospital in HOSPITALS:
            query = f"SELECT * FROM `{self.project_id}.{DATASET_ID}.{hospital}_data` WHERE data_split = '{train_split}'"
            df_train = self.client.query(query).to_dataframe()
            all_train_data.append(df_train)
            logger.info(f"  {hospital}: {len(df_train):,} training samples")

        # Combine all training data
        df_train_combined = pd.concat(all_train_data, ignore_index=True)
        logger.info(
            f"Combined training data: {len(df_train_combined):,} samples")

        # Fit preprocessor on TRAINING data only
        # This learns statistics that will be applied to test data
        self.preprocessor.fit(df_train_combined)

        logger.info("Preprocessor fitted successfully")
        logger.info("  Scaler learned mean/std from training data")
        logger.info("  Encoders learned category mappings from training data")
        logger.info("  Test data will be transformed using these statistics")
        logger.info("="*70)

    def load_test_data(self, hospital: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load and preprocess test data for specific hospital.

        Args:
            hospital: Hospital name (hospital_a, hospital_b, or hospital_c)

        Returns:
            Tuple of (df_raw, X_processed, y_true) where:
                - df_raw: Original dataframe (for additional analysis if needed)
                - X_processed: Preprocessed features using TRAINING statistics
                - y_true: Binary mortality labels

        Note:
            Uses preprocessor fitted on training data - no fitting on test data.
        """
        logger.info(f"Loading test data for {hospital}")

        # Query test data from BigQuery
        test_split = get_config_value('tool.himas.data.test-split', 'test')
        query = f"SELECT * FROM `{self.project_id}.{DATASET_ID}.{hospital}_data` WHERE data_split = '{test_split}'"
        df_test = self.client.query(query).to_dataframe()

        logger.info(f"  Samples: {len(df_test):,}")
        logger.info(f"  Mortality: {df_test[TARGET].mean():.2%}")

        # Transform using FITTED preprocessor (critical: no fitting on test data)
        X_test, y_test = self.preprocessor.transform(df_test)

        return df_test, X_test, y_test

    def evaluate_hospital(self, hospital: str) -> Dict:
        """
        Evaluate model on single hospital's test data.

        Args:
            hospital: Hospital name

        Returns:
            Dictionary with comprehensive performance metrics including:
                - Basic: accuracy, precision, recall, F1, ROC AUC
                - Clinical: specificity, NPV, PPV
                - Confusion matrix: TP, TN, FP, FN
                - Sample statistics: n_samples, n_deaths, prevalence
        """
        logger.info(f"Evaluating {hospital}")

        # Load and preprocess test data
        df, X_test, y_test = self.load_test_data(hospital)

        # Generate predictions
        y_pred_proba = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        # Compute comprehensive metrics
        metrics = {
            'hospital': hospital,
            'threshold': self.threshold,
            'n_samples': len(y_test),
            'n_deaths': int(y_test.sum()),
            'prevalence': float(y_test.mean()),
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'average_precision': float(average_precision_score(y_test, y_pred_proba))
        }

        # Confusion matrix for clinical interpretation
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics['confusion_matrix'] = {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        metrics['specificity'] = float(
            tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        # Negative Predictive Value
        metrics['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

        logger.info(f"  Results: AUC={metrics['roc_auc']:.4f}, "
                    f"Recall={metrics['recall']:.2%}, Precision={metrics['precision']:.2%}")

        # Store for visualization
        self.results[hospital] = {
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        return metrics

    def evaluate_all_hospitals(self) -> List[Dict]:
        """
        Evaluate model across all hospitals and compute aggregated metrics.

        Returns:
            List of metric dictionaries:
                - Individual hospital metrics (3 dicts)
                - Aggregated metrics across all hospitals (1 dict)
        """
        logger.info("="*70)
        logger.info(f"Evaluating All Hospitals (Threshold: {self.threshold})")
        logger.info("="*70)

        # Evaluate each hospital
        hospital_metrics = [self.evaluate_hospital(h) for h in HOSPITALS]

        # Compute aggregated metrics across all hospitals
        logger.info("\nAggregating results across all hospitals")

        all_y_true = np.concatenate(
            [self.results[h]['y_test'] for h in HOSPITALS])
        all_y_pred = np.concatenate(
            [self.results[h]['y_pred'] for h in HOSPITALS])
        all_y_pred_proba = np.concatenate(
            [self.results[h]['y_pred_proba'] for h in HOSPITALS])

        # Aggregated metrics
        agg = {
            'hospital': 'AGGREGATED',
            'threshold': self.threshold,
            'n_samples': len(all_y_true),
            'n_deaths': int(all_y_true.sum()),
            'prevalence': float(all_y_true.mean()),
            'accuracy': float(accuracy_score(all_y_true, all_y_pred)),
            'precision': float(precision_score(all_y_true, all_y_pred, zero_division=0)),
            'recall': float(recall_score(all_y_true, all_y_pred, zero_division=0)),
            'f1_score': float(f1_score(all_y_true, all_y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(all_y_true, all_y_pred_proba)),
            'average_precision': float(average_precision_score(all_y_true, all_y_pred_proba))
        }

        # Aggregated confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        tn, fp, fn, tp = cm.ravel()

        agg['confusion_matrix'] = {'tn': int(tn), 'fp': int(
            fp), 'fn': int(fn), 'tp': int(tp)}
        agg['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        agg['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

        # Log aggregated results
        logger.info("="*70)
        logger.info(
            f"Aggregated Results: {agg['n_samples']:,} total test samples")
        logger.info(f"  ROC AUC: {agg['roc_auc']:.4f}")
        logger.info(
            f"  Recall: {agg['recall']:.2%} (caught {tp:,} of {agg['n_deaths']:,} deaths)")
        logger.info(
            f"  Precision: {agg['precision']:.2%} ({fp:,} false alarms)")
        logger.info(f"  Specificity: {agg['specificity']:.2%}")
        logger.info(f"  Deaths missed: {fn:,}")
        logger.info("="*70)

        hospital_metrics.append(agg)
        return hospital_metrics

    def generate_visualizations(self) -> None:
        """Generate 6 comprehensive visualization plots organized by model name."""
        logger.info("Generating visualizations")
        sns.set_style("whitegrid")

        # Create model-specific figures directory
        # e.g., "himas_federated_mortality_model_20251113_213246"
        model_name = self.model_path.stem
        figures_dir = self.output_dir / "figures" / model_name
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving figures to: {figures_dir}")

        # Generate all 6 visualizations
        self._plot_roc_curves(figures_dir)
        self._plot_precision_recall_curves(figures_dir)
        self._plot_confusion_matrices(figures_dir)
        self._plot_metrics_comparison(figures_dir)
        self._plot_prediction_distribution(figures_dir)
        self._plot_calibration_curves(figures_dir)

        logger.info("All 6 visualizations saved successfully")

    def _plot_roc_curves(self, save_dir: Path):
        """Plot ROC curves for all hospitals."""
        plt.figure(figsize=(10, 8))

        for hospital in HOSPITALS:
            data = self.results[hospital]
            fpr, tpr, _ = roc_curve(data['y_test'], data['y_pred_proba'])
            auc = roc_auc_score(data['y_test'], data['y_pred_proba'])
            plt.plot(
                fpr, tpr, label=f'{hospital.replace("_", " ").title()} (AUC={auc:.3f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('ROC Curves - ICU Mortality Prediction',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  [1/6] ROC curves saved")

    def _plot_precision_recall_curves(self, save_dir: Path):
        """Plot Precision-Recall curves for all hospitals."""
        plt.figure(figsize=(10, 8))

        for hospital in HOSPITALS:
            data = self.results[hospital]
            precision, recall, _ = precision_recall_curve(
                data['y_test'], data['y_pred_proba'])
            ap = average_precision_score(data['y_test'], data['y_pred_proba'])
            plt.plot(recall, precision,
                     label=f'{hospital.replace("_", " ").title()} (AP={ap:.3f})', linewidth=2)

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - ICU Mortality Prediction',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'precision_recall_curves.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  [2/6] Precision-Recall curves saved")

    def _plot_confusion_matrices(self, save_dir: Path):
        """Plot confusion matrices for all hospitals."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, hospital in enumerate(HOSPITALS):
            data = self.results[hospital]
            cm = confusion_matrix(data['y_test'], data['y_pred'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['Survived', 'Deceased'],
                        yticklabels=['Survived', 'Deceased'],
                        cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{hospital.replace("_", " ").title()}\n(Threshold: {self.threshold})',
                                fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)

        plt.tight_layout()
        plt.savefig(
            save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  [3/6] Confusion matrices saved")

    def _plot_metrics_comparison(self, save_dir: Path):
        """Plot comparison of key metrics across hospitals."""
        metrics_data = []
        for hospital in HOSPITALS:
            m = self.results[hospital]['metrics']
            metrics_data.append({
                'Hospital': hospital.replace('_', ' ').title(),
                'Accuracy': m['accuracy'],
                'Precision': m['precision'],
                'Recall': m['recall'],
                'F1': m['f1_score'],
                'AUC': m['roc_auc']
            })

        df_metrics = pd.DataFrame(metrics_data)
        df_melted = df_metrics.melt(
            id_vars='Hospital', var_name='Metric', value_name='Score')

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melted, x='Metric', y='Score',
                    hue='Hospital', palette='Set2')
        plt.title(
            f'Performance Metrics Comparison (Threshold: {self.threshold})', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Metric', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title='Hospital', fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(
            save_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  [4/6] Metrics comparison saved")

    def _plot_prediction_distribution(self, save_dir: Path):
        """Plot distribution of prediction probabilities."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, hospital in enumerate(HOSPITALS):
            data = self.results[hospital]
            proba_survived = data['y_pred_proba'][data['y_test'] == 0]
            proba_deceased = data['y_pred_proba'][data['y_test'] == 1]

            axes[idx].hist(proba_survived, bins=30, alpha=0.6, label='Survived (True)',
                           color='green', density=True)
            axes[idx].hist(proba_deceased, bins=30, alpha=0.6, label='Deceased (True)',
                           color='red', density=True)
            axes[idx].axvline(x=self.threshold, color='black', linestyle='--',
                              linewidth=2, label=f'Threshold ({self.threshold})')
            axes[idx].set_title(
                f'{hospital.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(
                'Predicted Mortality Probability', fontsize=10)
            axes[idx].set_ylabel('Density', fontsize=10)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            save_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  [5/6] Prediction distribution saved")

    def _plot_calibration_curves(self, save_dir: Path):
        """Plot calibration curves showing predicted vs actual mortality rates."""
        from sklearn.calibration import calibration_curve

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, hospital in enumerate(HOSPITALS):
            data = self.results[hospital]

            # Compute calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                data['y_test'], data['y_pred_proba'], n_bins=10, strategy='uniform'
            )

            # Plot calibration curve
            axes[idx].plot(mean_predicted_value, fraction_of_positives, 's-',
                           linewidth=2, markersize=8, label='Model')
            axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=1,
                           label='Perfect Calibration')

            axes[idx].set_title(
                f'{hospital.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Mean Predicted Probability', fontsize=10)
            axes[idx].set_ylabel(
                'Fraction of Positives (True Mortality)', fontsize=10)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim([0, 1])
            axes[idx].set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(save_dir / 'calibration_curves.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("  [6/6] Calibration curves saved")

    def save_results(self, metrics: List[Dict]) -> None:
        """
        Save evaluation results in JSON format with complete metadata.

        Saves comprehensive results package including:
        - Evaluation timestamp
        - Model and hyperparameters paths
        - Preprocessing methodology (documents data leakage prevention)
        - Prediction threshold used
        - Hospital-specific metrics
        - Aggregated metrics

        Args:
            metrics: List of metric dictionaries from evaluate_all_hospitals()

        Output:
            JSON file: evaluation_results/metrics_{timestamp}.json
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Assemble comprehensive results package
        results_package = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'prediction_threshold': self.threshold,
            'hyperparameters_path': get_config_value('tool.flwr.app.config.shared-hyperparameters'),
            'hyperparameters': self.hyperparameters,
            'configuration': {
                'project_id': PROJECT_ID,
                'dataset_id': DATASET_ID,
                'hospitals': HOSPITALS
            },
            'metrics_by_hospital': metrics
        }

        # Save JSON
        json_path = self.output_dir / "results" / f'evaluation_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results_package, f, indent=2)

        logger.info(f"Results saved to {json_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for threshold configuration.

    Returns:
        Namespace with threshold argument

    Example:
        python evaluate_model.py --threshold 0.45
    """
    parser = argparse.ArgumentParser(
        description='Evaluate HIMAS federated model with configurable threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use default threshold from config
  %(prog)s --threshold 0.45         # Use custom threshold
  %(prog)s --threshold 0.467        # Use optimized threshold from optimization
        """
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Prediction threshold (0.0-1.0). Default: from pyproject.toml or 0.5'
    )

    return parser.parse_args()


def main():
    """
    Execute complete model evaluation workflow.

    Workflow:
    1. Parse command-line arguments
    2. Load model and configuration
    3. Fit preprocessor on training data (prevents leakage)
    4. Evaluate on test data from all hospitals
    5. Generate visualizations
    6. Save comprehensive results
    """
    args = parse_args()

    # Determine threshold from args, config, or default
    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"Using threshold from command line: {threshold}")
    else:
        threshold = get_config_value(
            'tool.himas.model.prediction-threshold', 0.5)
        logger.info(f"Using threshold from configuration: {threshold}")

    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Threshold must be between 0.0 and 1.0, got {threshold}")

    # Determine latest model path at runtime (after training has produced models)
    model_path = get_latest_model_path()

    logger.info("=" * 70)
    logger.info("HIMAS FEDERATED MODEL EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Model: {model_path}")
    logger.info(f"Threshold: {threshold}")
    logger.info("=" * 70)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"evaluation_{run_ts}") as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")

        # --- NEW: inspect tracking + artifact URIs for debugging ---
        client = MlflowClient()
        run_info = client.get_run(run.info.run_id).info
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Artifact URI for this run: {run_info.artifact_uri}")

        # Basic tags and params for the evaluation
        mlflow.set_tags({"role": "evaluator", "phase": "evaluation"})
        mlflow.log_params(
            {
                "evaluation_threshold": float(threshold),
                "model_path": str(model_path),
                "project_id": str(PROJECT_ID),
                "dataset_id": str(DATASET_ID),
            }
        )

        # ---------------- Core evaluation workflow ----------------
        evaluator = ModelEvaluator(model_path, PROJECT_ID, threshold)

        evaluator.load_model_and_config()
        evaluator.load_model_and_config()

        # CRITICAL: Fit preprocessor on training data first
        evaluator.load_model_and_config()

        # CRITICAL: Fit preprocessor on training data first
        evaluator.fit_preprocessor_on_training_data()
        evaluator.fit_preprocessor_on_training_data()

        # Evaluate on test data (using fitted preprocessor)
        evaluator.fit_preprocessor_on_training_data()

        # Evaluate on test data (using fitted preprocessor)
        metrics = evaluator.evaluate_all_hospitals()
        metrics = evaluator.evaluate_all_hospitals()

        # Generate visualizations
        metrics = evaluator.evaluate_all_hospitals()

        # Generate visualizations
        evaluator.generate_visualizations()
        evaluator.generate_visualizations()

        # Save results
        evaluator.generate_visualizations()

        # Save results
        evaluator.save_results(metrics)

        # ---------------- Log metrics to MLflow ----------------
        numeric_keys = {
            "n_samples",
            "n_deaths",
            "prevalence",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "average_precision",
            "specificity",
            "npv",
        }

        for m in metrics:
            prefix = str(m.get("hospital", "unknown")).lower()
            for k in numeric_keys:
                if k in m and isinstance(m[k], (int, float)):
                    mlflow.log_metric(f"{prefix}_{k}", float(m[k]))
            if "threshold" in m:
                mlflow.log_metric(f"{prefix}_threshold", float(m["threshold"]))

        # --- NEW: small debug artifact so we always see something ---
        mlflow.log_text("hello from evaluator", "debug_hello.txt")

        # ---------------- Log artifacts to MLflow ----------------
        figs_dir = evaluator.output_dir / "figures"
        res_dir = evaluator.output_dir / "results"

        if figs_dir.exists():
            logger.info(f"Logging figures from {figs_dir} to MLflow")
            mlflow.log_artifacts(str(figs_dir), artifact_path="figures")
        else:
            logger.warning(f"Figures directory not found: {figs_dir}")

        if res_dir.exists():
            logger.info(f"Logging results from {res_dir} to MLflow")
            mlflow.log_artifacts(str(res_dir), artifact_path="results")
        else:
            logger.warning(f"Results directory not found: {res_dir}")

        logger.info(f"Finished MLflow run: {run.info.run_id}")

    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY")
    logger.info(f"Results directory: {evaluator.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
