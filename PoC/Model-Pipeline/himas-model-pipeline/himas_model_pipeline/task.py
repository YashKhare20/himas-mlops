"""
HIMAS Task Module - ICU Mortality Prediction
===============================================

This module provides core functionality for federated learning including:
- Data loading from BigQuery with proper train/validation/test splits
- Feature preprocessing with hospital-specific StandardScalers (no data leakage)
- Model architecture construction with reproducible seeding
- Hyperparameter loading and configuration management
"""

import os
import logging
import random
import numpy as np
import pandas as pd
import json
import toml
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import tensorflow as tf
import keras
from keras import layers
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Global state management
_hospital_preprocessors = {}  # Store separate preprocessor per hospital
_feature_dim = None  # Cached feature dimension
_config_cache = None  # Cached pyproject.toml configuration


def set_seed(seed: int = 42) -> None:
    """
    Configure deterministic behavior across all random number generators.

    Critical for reproducibility in healthcare AI where regulatory compliance
    and validation require identical results across training runs.

    Args:
        seed: Random seed value for all libraries (Python, NumPy, TensorFlow)

    Note:
        Enables TensorFlow deterministic operations which may impact performance
        but ensures exact reproducibility across hardware and software configurations.
    """
    logger.info(f"Setting random seed: {seed}")

    # Python built-in random
    random.seed(seed)

    # NumPy random number generator
    np.random.seed(seed)

    # TensorFlow random operations
    tf.random.set_seed(seed)

    # Hash-based operations (Python sets, dicts)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # TensorFlow deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.experimental.enable_op_determinism()

    logger.debug("Random seeds configured for reproducibility")


def load_config() -> Dict[str, Any]:
    """
    Load configuration from pyproject.toml file.

    Searches current directory and all parent directories for pyproject.toml,
    caches the configuration for subsequent calls to avoid repeated file I/O.

    Returns:
        Dictionary containing complete TOML configuration

    Raises:
        FileNotFoundError: If pyproject.toml not found in any parent directory
    """
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    current = Path.cwd()
    for parent in [current] + list(current.parents):
        pyproject_path = parent / 'pyproject.toml'
        if pyproject_path.exists():
            logger.debug(f"Loading configuration from {pyproject_path}")
            with open(pyproject_path, 'r') as f:
                config = toml.load(f)
            _config_cache = config
            logger.info("Configuration loaded successfully")
            return config

    raise FileNotFoundError(
        "pyproject.toml not found in current or parent directories")


def get_config_value(key_path: str, default=None) -> Any:
    """
    Retrieve configuration value using dot notation path.

    Args:
        key_path: Dot-separated path to configuration value (e.g., 'tool.himas.data.project-id')
        default: Default value if key path not found

    Returns:
        Configuration value or default if not found

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


class DataPreprocessor:
    """
    Healthcare data preprocessing with protection against data leakage.

    Implements scikit-learn best practices for train/test preprocessing:
    - fit() on training data only to learn statistics
    - transform() on validation/test data using training statistics

    This approach ensures test data characteristics don't influence preprocessing
    parameters, maintaining valid performance estimates for production deployment.

    Attributes:
        numerical_scaler: StandardScaler fitted on training numerical features
        label_encoders: Dictionary of LabelEncoders fitted on training categorical features
        is_fitted: Boolean indicating if preprocessor has been fitted
        feature_dim: Dimensionality of processed feature vectors
    """

    def __init__(self):
        """Initialize preprocessor with unfitted transformers."""
        self.numerical_scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False
        self.feature_dim = None
        logger.debug("DataPreprocessor initialized")

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor on training data and return transformed features.

        This method should ONLY be called on training data. It learns:
        - Mean and standard deviation for numerical feature scaling
        - Category mappings for categorical feature encoding

        Args:
            df: Training dataframe from BigQuery

        Returns:
            Tuple of (X_processed, y) where:
                - X_processed: Scaled and encoded feature matrix (n_samples, n_features)
                - y: Binary mortality labels (n_samples,)
        """
        logger.info(f"Fitting preprocessor on {len(df):,} samples")

        # Load feature configuration
        target = get_config_value(
            'tool.himas.data.target-column', 'icu_mortality_label')
        excluded = get_config_value('tool.himas.data.excluded-columns')
        numerical_features = get_config_value(
            'tool.himas.data.numerical-features')
        categorical_features = get_config_value(
            'tool.himas.data.categorical-features')

        # Separate features from target
        exclude_cols = [target] + excluded
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[target].values

        # Process numerical features: convert to float64, impute, scale
        logger.debug(
            f"Processing {len(numerical_features)} numerical features")
        X_numerical = X[numerical_features].copy()
        for col in numerical_features:
            if col in X_numerical.columns:
                X_numerical[col] = pd.to_numeric(
                    X_numerical[col], errors='coerce').astype('float64')
        X_numerical = X_numerical.fillna(X_numerical.median())
        X_numerical_scaled = self.numerical_scaler.fit_transform(X_numerical)

        # Process categorical features: encode to integers
        logger.debug(
            f"Processing {len(categorical_features)} categorical features")
        X_categorical_encoded = []
        for col in categorical_features:
            if col in X.columns:
                self.label_encoders[col] = LabelEncoder()
                col_data = X[col].fillna('Unknown').astype(str)
                encoded = self.label_encoders[col].fit_transform(col_data)
                X_categorical_encoded.append(encoded.reshape(-1, 1))
                logger.debug(
                    f"  {col}: {len(self.label_encoders[col].classes_)} categories")

        # Combine numerical and categorical features
        if X_categorical_encoded:
            X_categorical_array = np.hstack(X_categorical_encoded)
            X_processed = np.hstack([X_numerical_scaled, X_categorical_array])
        else:
            X_processed = X_numerical_scaled

        self.is_fitted = True
        self.feature_dim = X_processed.shape[1]

        # Log preprocessing statistics
        logger.info(f"Preprocessor fitted successfully")
        logger.info(f"  Feature dimension: {self.feature_dim}")
        logger.info(f"  Mortality rate: {y.mean():.2%}")
        logger.info(f"  Positive samples: {y.sum():,} ({y.mean():.2%})")
        logger.info(
            f"  Negative samples: {(len(y) - y.sum()):,} ({(1-y.mean()):.2%})")

        return X_processed, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted preprocessor (NO fitting on validation/test data).

        Uses statistics learned from training data to transform new data.
        This prevents data leakage and ensures valid performance estimates.

        Args:
            df: Validation or test dataframe from BigQuery

        Returns:
            Tuple of (X_processed, y) with same shape as fit_transform output

        Raises:
            ValueError: If called before fit_transform()
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        logger.debug(f"Transforming {len(df):,} samples")

        # Load configuration
        target = get_config_value(
            'tool.himas.data.target-column', 'icu_mortality_label')
        excluded = get_config_value('tool.himas.data.excluded-columns')
        numerical_features = get_config_value(
            'tool.himas.data.numerical-features')
        categorical_features = get_config_value(
            'tool.himas.data.categorical-features')

        # Separate features from target
        exclude_cols = [target] + excluded
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[target].values

        # Transform numerical features (uses TRAINING mean/std)
        X_numerical = X[numerical_features].copy()
        for col in numerical_features:
            if col in X_numerical.columns:
                X_numerical[col] = pd.to_numeric(
                    X_numerical[col], errors='coerce').astype('float64')
        X_numerical = X_numerical.fillna(X_numerical.median())
        X_numerical_scaled = self.numerical_scaler.transform(
            X_numerical)  # Transform only, no fit

        # Transform categorical features (uses TRAINING encodings)
        X_categorical_encoded = []
        for col in categorical_features:
            if col in X.columns and col in self.label_encoders:
                col_data = X[col].fillna('Unknown').astype(str)
                le = self.label_encoders[col]
                # Handle categories unseen in training data
                encoded = np.array(
                    [le.transform([val])[0] if val in le.classes_ else -1 for val in col_data])
                X_categorical_encoded.append(encoded.reshape(-1, 1))

        # Combine features
        if X_categorical_encoded:
            X_categorical_array = np.hstack(X_categorical_encoded)
            X_processed = np.hstack([X_numerical_scaled, X_categorical_array])
        else:
            X_processed = X_numerical_scaled

        logger.debug(f"Transform complete - shape: {X_processed.shape}")
        return X_processed, y


def load_model(
    input_dim: int,
    hyperparameters: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None
) -> keras.Model:
    """
    Construct Multi-Layer Perceptron for ICU mortality prediction.

    Builds a deep neural network with configurable architecture optimized for
    tabular healthcare data. Supports three architecture patterns:
    - decreasing: Layer sizes halve progressively (256 → 128 → 64 → 32)
    - uniform: All hidden layers same size
    - bottleneck: Narrow in middle, wider at ends

    Args:
        input_dim: Number of input features after preprocessing
        hyperparameters: Optional dict with tuned hyperparameters from Colab optimization
        seed: Random seed for reproducible weight initialization

    Returns:
        Compiled Keras model ready for training

    Note:
        All hospitals must use identical architectures for federated learning.
        Weight aggregation fails if model structures differ across clients.
    """

    if seed is not None:
        set_seed(seed)

    # Load hyperparameters from file or use defaults
    if hyperparameters is None:
        logger.info("Using default hyperparameters from pyproject.toml")
        hp = {
            'num_layers': get_config_value('tool.himas.model.num-layers', 4),
            'architecture': get_config_value('tool.himas.model.architecture', 'decreasing'),
            'first_layer_units': get_config_value('tool.himas.model.first-layer-units', 256),
            'activation': get_config_value('tool.himas.model.activation', 'relu'),
            'dropout_rate': get_config_value('tool.himas.model.dropout-rate', 0.3),
            'l2_strength': get_config_value('tool.himas.model.l2-strength', 0.001),
            'learning_rate': get_config_value('tool.himas.model.learning-rate', 0.001),
            'optimizer': get_config_value('tool.himas.model.optimizer', 'adam')
        }
    else:
        logger.info("Using tuned hyperparameters from Colab optimization")
        hp = hyperparameters

    # Log architecture configuration
    logger.info("Building model architecture:")
    logger.info(f"  Type: {hp['architecture']}")
    logger.info(f"  Layers: {hp['num_layers']}")
    logger.info(f"  First layer units: {hp['first_layer_units']}")
    logger.info(f"  Activation: {hp['activation']}")
    logger.info(f"  Dropout rate: {hp['dropout_rate']}")
    logger.info(f"  L2 regularization: {hp['l2_strength']}")

    # Build sequential model
    model = keras.Sequential([keras.Input(shape=(input_dim,))])

    # Construct layers based on architecture type
    if hp['architecture'] == 'decreasing':
        # Progressively decreasing layer sizes (common for tabular data)
        units = hp['first_layer_units']
        for i in range(hp['num_layers']):
            model.add(layers.Dense(
                units,
                activation=hp['activation'],
                kernel_regularizer=keras.regularizers.l2(hp['l2_strength']),
                kernel_initializer=keras.initializers.GlorotUniform(
                    seed=seed if seed else None),
                name=f'dense_{i}'
            ))
            model.add(layers.BatchNormalization(name=f'bn_{i}'))
            model.add(layers.Dropout(
                hp['dropout_rate'], seed=seed if seed else None, name=f'dropout_{i}'))
            units = max(32, units // 2)  # Halve units, minimum 32

    elif hp['architecture'] == 'uniform':
        # Uniform layer sizes (stable capacity throughout)
        for i in range(hp['num_layers']):
            model.add(layers.Dense(
                hp['first_layer_units'],
                activation=hp['activation'],
                kernel_regularizer=keras.regularizers.l2(hp['l2_strength']),
                kernel_initializer=keras.initializers.GlorotUniform(
                    seed=seed if seed else None),
                name=f'dense_{i}'
            ))
            model.add(layers.BatchNormalization(name=f'bn_{i}'))
            model.add(layers.Dropout(
                hp['dropout_rate'], seed=seed if seed else None, name=f'dropout_{i}'))

    else:  # bottleneck architecture
        # Narrow in middle, wider at ends (dimension reduction then expansion)
        units_sequence = [
            max(32, hp['first_layer_units'] // (2 ** i) if i < hp['num_layers'] // 2
                else hp['first_layer_units'] // (2 ** (hp['num_layers'] - i - 1)))
            for i in range(hp['num_layers'])
        ]
        for i, units in enumerate(units_sequence):
            model.add(layers.Dense(
                units,
                activation=hp['activation'],
                kernel_regularizer=keras.regularizers.l2(hp['l2_strength']),
                kernel_initializer=keras.initializers.GlorotUniform(
                    seed=seed if seed else None),
                name=f'dense_{i}'
            ))
            model.add(layers.BatchNormalization(name=f'bn_{i}'))
            model.add(layers.Dropout(
                hp['dropout_rate'], seed=seed if seed else None, name=f'dropout_{i}'))

    # Output layer: single sigmoid unit for binary classification
    model.add(layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=keras.initializers.GlorotUniform(
            seed=seed if seed else None),
        name='output'
    ))

    # Configure optimizer
    optimizer_map = {
        'adam': keras.optimizers.Adam,
        'adamw': keras.optimizers.AdamW,
        'rmsprop': keras.optimizers.RMSprop
    }
    optimizer = optimizer_map[hp['optimizer']](
        learning_rate=hp['learning_rate'])

    # Compile with binary cross-entropy and healthcare-specific metrics
    # AUC is critical for imbalanced medical data
    # Precision and recall track clinical safety
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    total_params = model.count_params()
    logger.info("Model compiled successfully:")
    logger.info(f"  Optimizer: {hp['optimizer']}")
    logger.info(f"  Learning rate: {hp['learning_rate']:.6f}")
    logger.info(f"  Total parameters: {total_params:,}")

    return model


def load_data_from_bigquery(
    partition_id: int,
    num_partitions: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess hospital data for federated learning.

    IMPORTANT: Each hospital maintains its own preprocessor fitted on its own
    training data. This preserves data privacy and reflects the federated learning
    paradigm where hospitals don't share raw data statistics.

    Args:
        partition_id: Hospital identifier (0=Hospital A, 1=Hospital B, 2=Hospital C)
        num_partitions: Total number of hospitals (should be 3 for HIMAS)

    Returns:
        Tuple of (x_train, y_train, x_val, y_val) as numpy arrays

    Note:
        First call for each hospital creates and fits a new preprocessor.
        Subsequent calls reuse the fitted preprocessor for consistency within
        each hospital's training process across federated rounds.
    """
    global _hospital_preprocessors, _feature_dim

    # Load configuration
    project_id = get_config_value('tool.himas.data.project-id')
    dataset_id = get_config_value('tool.himas.data.dataset-id')
    hospital_names = get_config_value(
        'tool.himas.data.hospital-names', ['hospital_a', 'hospital_b', 'hospital_c'])
    train_split = get_config_value('tool.himas.data.train-split', 'train')
    val_split = get_config_value(
        'tool.himas.data.validation-split', 'validation')

    hospital_name = hospital_names[partition_id]
    table_name = f"{hospital_name}_data"

    logger.info("="*70)
    logger.info(
        f"Data Loading: {hospital_name.upper()} (Partition {partition_id})")
    logger.info("="*70)

    # Query BigQuery for hospital data
    client = bigquery.Client(project=project_id)

    df_train = client.query(
        f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}` WHERE data_split = '{train_split}'"
    ).to_dataframe()

    df_val = client.query(
        f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}` WHERE data_split = '{val_split}'"
    ).to_dataframe()

    target = get_config_value(
        'tool.himas.data.target-column', 'icu_mortality_label')
    logger.info(f"Loaded {len(df_train):,} train, {len(df_val):,} val samples")
    logger.info(f"Mortality: {df_train[target].mean():.2%}")

    # Create hospital-specific preprocessor (prevents cross-hospital data leakage)
    if partition_id not in _hospital_preprocessors:
        logger.info(f"Creating NEW preprocessor for {hospital_name}")
        preprocessor = DataPreprocessor()
        x_train, y_train = preprocessor.fit_transform(df_train)
        _hospital_preprocessors[partition_id] = preprocessor

        # Cache feature dimension globally
        if _feature_dim is None:
            _feature_dim = preprocessor.feature_dim
    else:
        logger.info(f"Reusing existing preprocessor for {hospital_name}")
        preprocessor = _hospital_preprocessors[partition_id]
        x_train, y_train = preprocessor.transform(df_train)

    # Transform validation data using hospital's fitted preprocessor
    x_val, y_val = preprocessor.transform(df_val)

    logger.info(
        f"Preprocessing complete: train={x_train.shape}, val={x_val.shape}")
    logger.info("="*70)

    return x_train, y_train, x_val, y_val


def get_feature_dim() -> int:
    """
    Retrieve cached feature dimension.

    Returns:
        Number of features after preprocessing

    Raises:
        ValueError: If data hasn't been loaded yet (dimension not set)
    """
    global _feature_dim
    if _feature_dim is None:
        raise ValueError("Feature dimension not set - load data first")
    return _feature_dim


def load_hyperparameters(json_path: str) -> Dict[str, Any]:
    """
    Load hyperparameters from JSON file generated by Colab tuning.

    Args:
        json_path: Path to hyperparameters JSON file

    Returns:
        Dictionary of hyperparameter values

    Raises:
        FileNotFoundError: If JSON file doesn't exist
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {json_path}")

    logger.info(f"Loading hyperparameters from {json_path}")
    with open(json_path, 'r') as f:
        hyperparameters = json.load(f)

    logger.debug(f"Loaded {len(hyperparameters)} hyperparameter values")
    return hyperparameters


def get_shared_hyperparameters(context=None) -> Optional[Dict[str, Any]]:
    """
    Load shared hyperparameters for federated learning.

    In federated learning, ALL hospitals must use identical model architectures
    to enable weight aggregation. This function loads the single set of optimized
    hyperparameters discovered through Colab tuning.

    Args:
        context: Optional Flower Context object from client/server

    Returns:
        Dictionary of hyperparameters or None if not found (falls back to defaults)

    Note:
        Checks context.run_config first, then pyproject.toml configuration
    """

    # Try to get from Flower context (runtime configuration)
    if context and hasattr(context, 'run_config'):
        hp_path = context.run_config.get('shared-hyperparameters')
        if hp_path and Path(hp_path).exists():
            logger.info(
                f"Loading shared hyperparameters from context: {hp_path}")
            return load_hyperparameters(hp_path)

    # Fall back to pyproject.toml configuration
    hp_path = get_config_value('tool.flwr.app.config.shared-hyperparameters')
    if hp_path and Path(hp_path).exists():
        logger.info(
            f"Loading shared hyperparameters from pyproject.toml: {hp_path}")
        return load_hyperparameters(hp_path)

    logger.warning(
        "No shared hyperparameters found - using default configuration")
    return None
