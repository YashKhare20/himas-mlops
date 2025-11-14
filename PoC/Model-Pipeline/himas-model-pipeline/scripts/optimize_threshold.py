"""
HIMAS Threshold Optimization System
====================================

Analyzes prediction thresholds for clinical deployment of ICU mortality models.

Key Clinical Considerations:
----------------------------
- False Negatives (missed deaths) have severe consequences
- False Positives (false alarms) cause alert fatigue and resource waste
- Different thresholds serve different clinical scenarios (screening vs intervention)
- Cost-sensitive optimization weighs FN/FP based on clinical priorities

Output Artifacts:
-----------------
- CSV: Comprehensive threshold comparison table
- JSON: Machine-readable recommendations
- Markdown: Clinical implementation guide
- PNG: Threshold impact visualizations
"""

import logging
import numpy as np
import pandas as pd
import json
import toml
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import keras
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score
)

# Configure logging with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config() -> Dict:
    """
    Load TOML configuration from pyproject.toml.

    Searches current directory and all parent directories for pyproject.toml file.
    This allows the script to be run from any subdirectory within the project.

    Returns:
        Dictionary containing complete project configuration

    Raises:
        FileNotFoundError: If pyproject.toml not found in any parent directory
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        pyproject_path = parent / 'pyproject.toml'
        if pyproject_path.exists():
            with open(pyproject_path, 'r') as f:
                return toml.load(f)
    raise FileNotFoundError("pyproject.toml not found")


def get_config_value(key_path: str, default=None):
    """
    Retrieve nested configuration value using dot notation.

    Args:
        key_path: Dot-separated path (e.g., 'tool.himas.data.project-id')
        default: Fallback value if key not found

    Returns:
        Configuration value or default

    Example:
        >>> project_id = get_config_value('tool.himas.data.project-id')
    """
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


# Load configuration constants (loaded once at module import)
PROJECT_ID = get_config_value('tool.himas.data.project-id')
DATASET_ID = get_config_value('tool.himas.data.dataset-id', 'federated')
MODEL_PATH = get_latest_model_path()
HOSPITALS = get_config_value(
    'tool.himas.data.hospital-names', ['hospital_a', 'hospital_b', 'hospital_c'])
NUMERICAL_FEATURES = get_config_value('tool.himas.data.numerical-features')
CATEGORICAL_FEATURES = get_config_value('tool.himas.data.categorical-features')
TARGET = get_config_value(
    'tool.himas.data.target-column', 'icu_mortality_label')


class DataPreprocessor:
    """
    Scikit-learn compliant data preprocessor preventing test data leakage.

    Implements the correct preprocessing pattern:
    1. fit() on training data to learn statistics (mean, std, categories)
    2. transform() on test data using learned statistics

    This ensures test data characteristics don't influence preprocessing parameters,
    maintaining valid performance estimates critical for healthcare model validation
    and regulatory approval.

    Attributes:
        numerical_scaler: StandardScaler for z-score normalization
        label_encoders: Dict mapping categorical column names to fitted LabelEncoders
        is_fitted: Boolean flag indicating if fit() has been called

    Reference:
        Scikit-learn best practices: https://scikit-learn.org/stable/common_pitfalls.html
    """

    def __init__(self):
        """Initialize unfitted preprocessor."""
        self.numerical_scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Fit preprocessing transformations on training data exclusively.

        Learns the following from training data:
        - Mean and standard deviation for each numerical feature
        - Category-to-integer mappings for each categorical feature

        Args:
            df_train: Training dataframe containing all features and target

        Note:
            This method must NEVER be called on validation or test data.
            Doing so would cause data leakage and invalidate performance metrics.
        """
        logger.info(
            f"Fitting preprocessor on {len(df_train):,} TRAINING samples")

        # Separate features from excluded columns and target
        excluded = get_config_value('tool.himas.data.excluded-columns', [])
        exclude_cols = [TARGET] + excluded
        X_train = df_train.drop(
            columns=[col for col in exclude_cols if col in df_train.columns])

        # Fit numerical scaler using TRAINING data statistics
        X_numerical = X_train[NUMERICAL_FEATURES].copy()
        for col in NUMERICAL_FEATURES:
            if col in X_numerical.columns:
                # Convert BigQuery Int64 to float64 to avoid dtype issues
                X_numerical[col] = pd.to_numeric(
                    X_numerical[col], errors='coerce').astype('float64')

        # Impute missing values with median (computed from training data)
        X_numerical = X_numerical.fillna(X_numerical.median())

        # Fit scaler: learns mean and std from training data
        self.numerical_scaler.fit(X_numerical)

        # Fit label encoders using TRAINING data categories
        for col in CATEGORICAL_FEATURES:
            if col in X_train.columns:
                self.label_encoders[col] = LabelEncoder()
                col_data = X_train[col].fillna('Unknown').astype(str)
                # Fit encoder: learns category-to-integer mapping from training data
                self.label_encoders[col].fit(col_data)

        self.is_fitted = True
        logger.info(
            "Preprocessor fitted on training data (no test data leakage)")

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using preprocessing fitted on training data.

        Applies the preprocessing transformations learned from training data
        to new data (validation or test). Uses training statistics (mean, std,
        category mappings) without fitting on the new data.

        Args:
            df: Dataframe to transform (validation or test data)

        Returns:
            Tuple of (X_processed, y) where:
                - X_processed: Preprocessed feature matrix
                - y: Target labels

        Raises:
            ValueError: If called before fit()

        Note:
            This is the CORRECT approach for test data preprocessing.
            Never call fit() or fit_transform() on test data.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Separate features from target
        excluded = get_config_value('tool.himas.data.excluded-columns', [])
        exclude_cols = [TARGET] + excluded
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[TARGET].values

        # Transform numerical features using TRAINING mean/std
        X_numerical = X[NUMERICAL_FEATURES].copy()
        for col in NUMERICAL_FEATURES:
            if col in X_numerical.columns:
                X_numerical[col] = pd.to_numeric(
                    X_numerical[col], errors='coerce').astype('float64')

        # Impute with test data median (acceptable as it doesn't affect training statistics)
        X_numerical = X_numerical.fillna(X_numerical.median())

        # CRITICAL: Use transform() not fit_transform() - applies training statistics
        X_numerical_scaled = self.numerical_scaler.transform(X_numerical)

        # Transform categorical features using TRAINING encodings
        X_categorical_encoded = []
        for col in CATEGORICAL_FEATURES:
            if col in X.columns and col in self.label_encoders:
                col_data = X[col].fillna('Unknown').astype(str)
                le = self.label_encoders[col]

                # Handle categories unseen in training data (rare but possible)
                # Assign -1 to unseen categories - model trained to handle this
                encoded = np.array([
                    le.transform([val])[0] if val in le.classes_ else -1
                    for val in col_data
                ])
                X_categorical_encoded.append(encoded.reshape(-1, 1))

        # Concatenate numerical and categorical features
        if X_categorical_encoded:
            X_categorical_array = np.hstack(X_categorical_encoded)
            X_processed = np.hstack([X_numerical_scaled, X_categorical_array])
        else:
            X_processed = X_numerical_scaled

        return X_processed, y


class ThresholdAnalyzer:
    """
    Analyzes and optimizes prediction thresholds for clinical deployment.

    Implements multiple threshold optimization strategies to balance
    clinical priorities:

    Methods:
    --------
    1. Youden's Index: Maximizes (Sensitivity + Specificity - 1)
       - Balanced performance
       - Good for general risk stratification

    2. F1 Maximization: Maximizes harmonic mean of precision and recall
       - Balances false positives and false negatives equally
       - Standard ML optimization criterion

    3. Target Recall: Achieves specified sensitivity (e.g., 85%)
       - Prioritizes catching high-risk patients
       - Accepts more false alarms for safety

    4. Cost-Sensitive: Minimizes weighted cost (FN_cost * FN + FP_cost * FP)
       - Reflects real clinical costs and priorities
       - Typically weights FN >> FP in healthcare

    Attributes:
        model_path: Path to trained Keras model
        project_id: GCP project ID for BigQuery access
        client: BigQuery client for data retrieval
        model: Loaded Keras model
        preprocessor: Fitted DataPreprocessor for consistent transformation
        output_dir: Directory for saving results and visualizations
    """

    def __init__(self, model_path: str, project_id: str):
        """
        Initialize threshold analyzer.

        Args:
            model_path: Path to trained federated learning model (.keras file)
            project_id: Google Cloud Platform project identifier
        """
        self.model_path = Path(model_path)
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.model = None
        self.preprocessor = None

        # Create output directory structure
        self.output_dir = Path("threshold_optimization_results")
        self.output_dir.mkdir(exist_ok=True)

        # Create model-specific figures directory
        model_name = self.model_path.stem
        self.figures_dir = self.output_dir / "figures" / model_name
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Create results directory for reports
        self.results_dir = self.output_dir / "results" / model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> None:
        """
        Load trained Keras model from disk.

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        logger.info(f"Loading model from {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = keras.models.load_model(str(self.model_path))
        logger.info(
            f"Model loaded - Parameters: {self.model.count_params():,}")

    def fit_preprocessor_on_training_data(self) -> None:
        """
        Fit preprocessor on combined training data from all hospitals.

        CRITICAL FOR PREVENTING DATA LEAKAGE:
        --------------------------------------
        This method fits the preprocessor (StandardScaler and LabelEncoders)
        exclusively on training data. The fitted transformations are then applied
        to test data via transform(), ensuring test data characteristics don't
        influence preprocessing parameters.

        Why combine hospitals for fitting:
        ----------------------------------
        - Provides consistent preprocessing across all test evaluations
        - Mirrors the federated learning approach where hospitals share model weights
        - Enables fair comparison of threshold performance across hospitals

        Note:
            This differs from federated TRAINING where each hospital uses its own
            preprocessor. For EVALUATION, we need consistent preprocessing to
            fairly compare performance across hospitals and thresholds.
        """
        logger.info("="*70)
        logger.info(
            "Fitting Preprocessor on Training Data (Preventing Leakage)")
        logger.info("="*70)

        self.preprocessor = DataPreprocessor()

        # Load training data from all three hospitals
        train_split = get_config_value('tool.himas.data.train-split', 'train')
        all_train_data = []

        for hospital in HOSPITALS:
            query = f"SELECT * FROM `{self.project_id}.{DATASET_ID}.{hospital}_data` WHERE data_split = '{train_split}'"
            df_train = self.client.query(query).to_dataframe()
            all_train_data.append(df_train)
            logger.info(f"  {hospital}: {len(df_train):,} training samples")

        # Combine training data from all hospitals
        df_train_combined = pd.concat(all_train_data, ignore_index=True)
        logger.info(
            f"Combined: {len(df_train_combined):,} total training samples")

        # Fit preprocessor on TRAINING data only
        # This learns: mean/std for numerical features, category mappings for categorical
        self.preprocessor.fit(df_train_combined)

        logger.info("Preprocessor fitted using ONLY training statistics")
        logger.info("  Test data will be transformed using these statistics")
        logger.info("="*70)

    def load_and_predict_all_hospitals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data and generate predictions using fitted preprocessor.

        Loads test data from all hospitals, transforms it using the preprocessor
        fitted on training data, and generates mortality probability predictions.

        Returns:
            Tuple of (y_true, y_pred_proba) where:
                - y_true: Actual mortality labels (0=survived, 1=deceased)
                - y_pred_proba: Predicted mortality probabilities [0.0-1.0]

        Note:
            Combines all hospitals' test data for aggregate threshold optimization.
            Individual hospital performance can be analyzed separately if needed.
        """
        logger.info("="*70)
        logger.info("Loading Test Data and Generating Predictions")
        logger.info("="*70)

        all_y_true = []
        all_y_pred_proba = []

        test_split = get_config_value('tool.himas.data.test-split', 'test')

        for hospital in HOSPITALS:
            logger.info(f"Processing {hospital}")

            # Query test data from BigQuery
            query = f"SELECT * FROM `{self.project_id}.{DATASET_ID}.{hospital}_data` WHERE data_split = '{test_split}'"
            df_test = self.client.query(query).to_dataframe()
            logger.info(f"  Loaded {len(df_test):,} test samples")

            # Transform using FITTED preprocessor (no fitting on test data)
            # This applies training statistics - critical for preventing data leakage
            X_test, y_test = self.preprocessor.transform(df_test)

            # Generate predictions using trained model
            y_pred_proba = self.model.predict(X_test, verbose=0).flatten()

            all_y_true.append(y_test)
            all_y_pred_proba.append(y_pred_proba)

            logger.info(f"  Predictions generated")

        # Aggregate across all hospitals for comprehensive analysis
        y_true = np.concatenate(all_y_true)
        y_pred_proba = np.concatenate(all_y_pred_proba)

        logger.info(f"\nAggregated Statistics:")
        logger.info(f"  Total test samples: {len(y_true):,}")
        logger.info(f"  Mortality rate: {y_true.mean():.2%}")
        logger.info(f"  Total deaths: {y_true.sum():,}")
        logger.info("="*70)

        return y_true, y_pred_proba

    def optimize_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> pd.DataFrame:
        """
        Execute all threshold optimization strategies.

        Evaluates six different threshold selection methods to provide
        clinical stakeholders with options based on institutional priorities.

        Args:
            y_true: Actual mortality outcomes (binary labels)
            y_pred_proba: Predicted mortality probabilities

        Returns:
            DataFrame with columns: method, threshold, recall, precision, F1, etc.
            Sorted by method name for easy comparison

        Methods Evaluated:
        -----------------
        1. Youden's Index: Optimal sensitivity+specificity balance
        2. F1 Maximum: Optimal precision-recall balance
        3. Target Recall 85%: High sensitivity for screening
        4. Target Recall 80%: Moderate-high sensitivity
        5. Cost-Sensitive: Weighted by clinical costs (FN=10x FP)
        6. Current (0.5): Baseline for comparison
        """
        logger.info("="*70)
        logger.info("Running Threshold Optimization Methods")
        logger.info("="*70)

        results = []

        # Method 1: Youden's Index (J = Sensitivity + Specificity - 1)
        # Maximizes the vertical distance from ROC curve to diagonal
        logger.info("1. Youden's Index (maximizes sensitivity + specificity)")
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        # Sensitivity - (1 - Specificity) = Sensitivity + Specificity - 1
        youden_scores = tpr - fpr
        youden_idx = np.argmax(youden_scores)
        results.append(self._evaluate_threshold(
            y_true, y_pred_proba, roc_thresholds[youden_idx], "Youden Index"
        ))

        # Method 2: F1 Score Maximization
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        logger.info(
            "2. F1 Score Maximization (harmonic mean of precision/recall)")
        precision, recall, pr_thresholds = precision_recall_curve(
            y_true, y_pred_proba)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / \
            (precision[:-1] + recall[:-1] + 1e-10)
        f1_idx = np.argmax(f1_scores)
        results.append(self._evaluate_threshold(
            y_true, y_pred_proba, pr_thresholds[f1_idx], "F1 Maximum"
        ))

        # Method 3: Target Recall 85% (High Sensitivity)
        # For clinical scenarios where missing deaths is unacceptable
        logger.info("3. Target Recall 85% (high sensitivity for screening)")
        recall_idx_85 = np.argmin(np.abs(recall[:-1] - 0.85))
        results.append(self._evaluate_threshold(
            y_true, y_pred_proba, pr_thresholds[recall_idx_85], "Target Recall 85%"
        ))

        # Method 4: Target Recall 80% (Moderate-High Sensitivity)
        logger.info("4. Target Recall 80% (moderate-high sensitivity)")
        recall_idx_80 = np.argmin(np.abs(recall[:-1] - 0.80))
        results.append(self._evaluate_threshold(
            y_true, y_pred_proba, pr_thresholds[recall_idx_80], "Target Recall 80%"
        ))

        # Method 5: Cost-Sensitive Optimization
        # Reflects real clinical costs: Missing a death (FN) is ~10x worse than false alarm (FP)
        logger.info("5. Cost-Sensitive Optimization (FN cost = 10x FP cost)")
        cost_threshold = self._cost_sensitive_threshold(
            y_true, y_pred_proba, fn_cost=10, fp_cost=1
        )
        results.append(self._evaluate_threshold(
            y_true, y_pred_proba, cost_threshold, "Cost-Sensitive (10:1)"
        ))

        # Method 6: Current Baseline (0.5)
        # Standard ML threshold for comparison
        logger.info("6. Current Threshold (0.5 baseline)")
        results.append(self._evaluate_threshold(
            y_true, y_pred_proba, 0.5, "Current (0.5)"
        ))

        logger.info("="*70)
        return pd.DataFrame(results)

    def _evaluate_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float,
        method: str
    ) -> Dict:
        """
        Compute comprehensive clinical metrics for a specific threshold.

        Args:
            y_true: Actual mortality outcomes
            y_pred_proba: Predicted probabilities
            threshold: Decision threshold to evaluate
            method: Name of optimization method (for logging)

        Returns:
            Dictionary containing:
                - Performance metrics: recall, precision, specificity, F1, NPV, accuracy
                - Confusion matrix counts: TP, TN, FP, FN
                - Clinical impact: deaths_caught, deaths_missed
                - Metadata: method name, threshold value
        """
        # Convert probabilities to binary predictions using threshold
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate core metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Assemble comprehensive metrics dictionary
        metrics = {
            'method': method,
            'threshold': float(threshold),
            'recall': float(recall),  # Sensitivity: TP / (TP + FN)
            'precision': float(precision),  # PPV: TP / (TP + FP)
            # TN / (TN + FP)
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'f1_score': float(f1),
            # Negative Predictive Value
            'npv': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
            'accuracy': float((tp + tn) / (tp + tn + fp + fn)),

            # Confusion matrix components
            'tp': int(tp),  # True Positives: Correctly predicted deaths
            'tn': int(tn),  # True Negatives: Correctly predicted survivals
            'fp': int(fp),  # False Positives: False alarms
            'fn': int(fn),  # False Negatives: Missed deaths (most critical)

            # Clinical interpretation
            'total_deaths': int(y_true.sum()),
            'deaths_caught': int(tp),
            'deaths_missed': int(fn)  # Most important metric for safety
        }

        logger.info(f"  {method}: threshold={threshold:.3f}, "
                    f"recall={recall:.2%}, precision={precision:.2%}, F1={f1:.3f}")

        return metrics

    def _cost_sensitive_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        fn_cost: float,
        fp_cost: float
    ) -> float:
        """
        Find threshold minimizing total weighted cost.

        Cost function: Total_Cost = (FN * fn_cost) + (FP * fp_cost)

        Args:
            y_true: Actual outcomes
            y_pred_proba: Predicted probabilities
            fn_cost: Cost of missing a death (false negative)
            fp_cost: Cost of false alarm (false positive)

        Returns:
            Optimal threshold minimizing total cost

        Clinical Rationale:
            In ICU mortality prediction, missing a high-risk patient (FN) has
            severe consequences (patient death without intervention), while a
            false alarm (FP) causes resource waste and alert fatigue but no
            direct patient harm. Typical cost ratios: FN = 5-20x FP cost.
        """
        # Scan threshold range in fine increments
        thresholds = np.linspace(0.1, 0.9, 100)
        min_cost = float('inf')
        optimal_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # Calculate weighted total cost
            cost = (fn * fn_cost) + (fp * fp_cost)

            # Track minimum cost threshold
            if cost < min_cost:
                min_cost = cost
                optimal_threshold = threshold

        return optimal_threshold

    def visualize_threshold_impact(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """
        Create comprehensive visualizations of threshold impact on clinical metrics.

        Generates two key visualizations:
        1. Performance metrics (recall, precision, F1, etc.) vs threshold
        2. Clinical errors (false negatives, false positives) vs threshold

        Args:
            y_true: Actual mortality outcomes
            y_pred_proba: Predicted probabilities

        Output:
            Saves high-resolution PNG figures to output_dir/figures/
        """
        logger.info("Generating threshold impact visualizations")

        # Scan thresholds from 0.05 to 0.95 in fine increments
        thresholds = np.linspace(0.05, 0.95, 100)
        metrics = {
            'threshold': [], 'recall': [], 'precision': [], 'f1': [],
            'specificity': [], 'npv': [], 'fn': [], 'fp': []
        }

        # Compute metrics at each threshold
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            metrics['threshold'].append(threshold)
            metrics['recall'].append(recall)
            metrics['precision'].append(precision)
            metrics['f1'].append(f1)
            metrics['specificity'].append(
                tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            metrics['npv'].append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
            metrics['fn'].append(fn)
            metrics['fp'].append(fp)

        # ============================================================
        # Visualization 1: Clinical Performance Metrics
        # ============================================================
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot key clinical metrics
        ax.plot(metrics['threshold'], metrics['recall'],
                label='Recall (Sensitivity)', linewidth=3, color='#e74c3c')
        ax.plot(metrics['threshold'], metrics['precision'],
                label='Precision (PPV)', linewidth=3, color='#3498db')
        ax.plot(metrics['threshold'], metrics['f1'],
                label='F1 Score', linewidth=3, color='#2ecc71')
        ax.plot(metrics['threshold'], metrics['specificity'],
                label='Specificity', linewidth=2.5, color='#9b59b6', linestyle='--')
        ax.plot(metrics['threshold'], metrics['npv'],
                label='NPV', linewidth=2, color='#f39c12', linestyle=':')

        # Mark current threshold
        ax.axvline(x=0.5, color='black', linestyle=':',
                   alpha=0.6, linewidth=2, label='Current (0.5)')

        # Mark 80% recall target (common clinical goal)
        ax.axhline(y=0.80, color='gray', linestyle=':', alpha=0.3, linewidth=1)

        ax.set_xlabel('Prediction Threshold', fontsize=13)
        ax.set_ylabel('Metric Score', fontsize=13)
        ax.set_title('Clinical Performance Metrics vs Prediction Threshold',
                     fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / 'threshold_metrics_impact.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        logger.info("Metrics impact chart saved")

        # ============================================================
        # Visualization 2: Clinical Errors (FN and FP)
        # ============================================================
        # Dual-axis plot showing false negatives and false positives
        # Critical for understanding clinical impact of threshold choice
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot false negatives (missed deaths) on left axis
        ax1.plot(metrics['threshold'], metrics['fn'],
                 label='False Negatives (Missed Deaths)',
                 linewidth=3, color='#c0392b', marker='o', markersize=3)
        ax1.set_xlabel('Prediction Threshold', fontsize=13)
        ax1.set_ylabel('False Negatives (Deaths Missed)',
                       fontsize=13, color='#c0392b')
        ax1.tick_params(axis='y', labelcolor='#c0392b')
        ax1.grid(True, alpha=0.3)

        # Plot false positives (false alarms) on right axis
        ax2 = ax1.twinx()
        ax2.plot(metrics['threshold'], metrics['fp'],
                 label='False Positives (False Alarms)',
                 linewidth=3, color='#2980b9', marker='s', markersize=3)
        ax2.set_ylabel('False Positives (False Alarms)',
                       fontsize=13, color='#2980b9')
        ax2.tick_params(axis='y', labelcolor='#2980b9')

        # Mark current threshold
        ax1.axvline(x=0.5, color='black', linestyle=':',
                    alpha=0.6, linewidth=2)

        plt.title('Clinical Errors vs Prediction Threshold',
                  fontsize=15, fontweight='bold')

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper center', fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / 'threshold_errors_impact.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        logger.info("Errors impact chart saved")

    def generate_recommendations(self, results_df: pd.DataFrame) -> Dict:
        """
        Generate clinical deployment recommendations based on optimization results.

        Args:
            results_df: DataFrame with threshold evaluation results

        Returns:
            Dictionary with recommendations for different clinical scenarios:
                - screening: High sensitivity for patient screening
                - balanced: Youden's Index for general use
                - specific: High precision for intervention triggers
                - current: Baseline performance (0.5 threshold)
        """
        logger.info("="*70)
        logger.info("Generating Clinical Recommendations")
        logger.info("="*70)

        # Extract recommendations for different clinical scenarios
        recommendations = {
            'screening': results_df[results_df['method'].str.contains('85%')].iloc[0].to_dict()
            if not results_df[results_df['method'].str.contains('85%')].empty else None,
            'balanced': results_df[results_df['method'] == 'Youden Index'].iloc[0].to_dict(),
            'specific': results_df[results_df['method'] == 'F1 Maximum'].iloc[0].to_dict(),
            'current': results_df[results_df['method'] == 'Current (0.5)'].iloc[0].to_dict()
        }

        # Log recommendations with clinical context
        if recommendations['screening']:
            r = recommendations['screening']
            logger.info(f"HIGH SENSITIVITY SCREENING:")
            logger.info(f"  Threshold: {r['threshold']:.3f}")
            logger.info(
                f"  Recall: {r['recall']:.2%}, Precision: {r['precision']:.2%}")
            logger.info(f"  Use case: ICU admission triage, early warning")

        r = recommendations['balanced']
        logger.info(f"\nBALANCED (RECOMMENDED FOR GENERAL USE):")
        logger.info(f"  Threshold: {r['threshold']:.3f}")
        logger.info(
            f"  Recall: {r['recall']:.2%}, Precision: {r['precision']:.2%}")
        logger.info(f"  Use case: General risk stratification")

        r = recommendations['current']
        logger.info(f"\nCURRENT BASELINE (0.5):")
        logger.info(
            f"  Recall: {r['recall']:.2%}, Precision: {r['precision']:.2%}")
        logger.info(f"  Missed deaths: {r['deaths_missed']:,}")

        logger.info("="*70)
        return recommendations

    def save_results(self, results_df: pd.DataFrame, recommendations: Dict) -> None:
        """
        Save optimization results in multiple formats.

        Saves:
            - CSV: Threshold comparison table for spreadsheet analysis
            - JSON: Machine-readable results for programmatic access
            - Markdown: Clinical implementation guide for stakeholders

        Args:
            results_df: DataFrame with all threshold evaluations
            recommendations: Dict with scenario-specific recommendations
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save CSV for easy viewing and analysis
        csv_path = self.results_dir / f'threshold_analysis_{timestamp}.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"CSV analysis saved to {csv_path}")

        # Save JSON for programmatic access
        json_path = self.results_dir / f'recommendations_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'preprocessing': 'Fitted on training data only (no leakage)',
                'total_test_samples': len(results_df) * int(results_df.iloc[0]['total_deaths'] / results_df.iloc[0]['recall']) if results_df.iloc[0]['recall'] > 0 else 0,
                'recommendations': recommendations,
                'all_methods': results_df.to_dict('records')
            }, f, indent=2)
        logger.info(f"JSON recommendations saved to {json_path}")

        # Generate markdown report for clinical stakeholders
        md_path = self.results_dir / f'recommendations_{timestamp}.md'
        self._create_markdown_report(md_path, results_df, recommendations)
        logger.info(f"Markdown report saved to {md_path}")

    def _create_markdown_report(
        self,
        md_path: Path,
        results_df: pd.DataFrame,
        recommendations: Dict
    ) -> None:
        """
        Generate stakeholder-friendly markdown report.

        Creates a clinical implementation guide with:
        - Threshold comparison table
        - Recommended threshold with rationale
        - Code examples for implementation

        Args:
            md_path: Output path for markdown file
            results_df: Threshold evaluation results
            recommendations: Clinical scenario recommendations
        """
        with open(md_path, 'w') as f:
            f.write("# HIMAS Threshold Optimization Report\n\n")
            f.write(
                f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Threshold Methods Comparison\n\n")
            f.write(
                "| Method | Threshold | Recall | Precision | F1 | FN (Missed) | FP (Alarms) |\n")
            f.write(
                "|--------|-----------|--------|-----------|-----|-------------|-------------|\n")

            for _, row in results_df.iterrows():
                f.write(f"| {row['method']} | {row['threshold']:.3f} | {row['recall']:.1%} | "
                        f"{row['precision']:.1%} | {row['f1_score']:.3f} | {row['fn']} | {row['fp']} |\n")

            f.write("\n## Recommended Threshold (Youden's Index)\n\n")
            r = recommendations['balanced']
            f.write(f"**{r['threshold']:.3f}**\n\n")
            f.write(
                f"- **Recall:** {r['recall']:.1%} (catches {r['deaths_caught']:,} of {r['total_deaths']:,} deaths)\n")
            f.write(
                f"- **Precision:** {r['precision']:.1%} ({r['fp']:,} false alarms)\n")
            f.write(f"- **F1 Score:** {r['f1_score']:.3f}\n\n")

            f.write("## Implementation\n\n")
            f.write("Update `pyproject.toml`:\n\n")
            f.write("```toml\n")
            f.write("[tool.himas.model]\n")
            f.write(f"prediction-threshold = {r['threshold']:.4f}\n")
            f.write("```\n")


def main():
    """
    Execute complete threshold optimization workflow.

    Steps:
    1. Load trained federated model
    2. Fit preprocessor on all training data
    3. Load and preprocess test data
    4. Generate predictions
    5. Optimize thresholds using multiple methods
    6. Create visualizations
    7. Generate recommendations
    8. Save results in multiple formats
    """
    logger.info("="*70)
    logger.info("HIMAS THRESHOLD OPTIMIZATION SYSTEM")
    logger.info("="*70)

    # Initialize analyzer
    analyzer = ThresholdAnalyzer(MODEL_PATH, PROJECT_ID)

    # Load model
    analyzer.load_model()

    # CRITICAL: Fit preprocessor on training data first (prevents data leakage)
    analyzer.fit_preprocessor_on_training_data()

    # Load test data and generate predictions (using fitted preprocessor)
    y_true, y_pred_proba = analyzer.load_and_predict_all_hospitals()

    # Run all threshold optimization methods
    results_df = analyzer.optimize_thresholds(y_true, y_pred_proba)

    # Generate visualizations showing threshold impact
    analyzer.visualize_threshold_impact(y_true, y_pred_proba)

    # Generate clinical recommendations
    recommendations = analyzer.generate_recommendations(results_df)

    # Save all results
    analyzer.save_results(results_df, recommendations)

    logger.info("="*70)
    logger.info("THRESHOLD OPTIMIZATION COMPLETED SUCCESSFULLY")
    logger.info(f"Results directory: {analyzer.output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
