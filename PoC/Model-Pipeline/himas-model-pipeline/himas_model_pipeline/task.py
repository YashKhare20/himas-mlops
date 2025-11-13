"""himas-model-pipeline: A Flower / TensorFlow app for ICU mortality prediction."""

import os
import numpy as np
import pandas as pd
import keras
from keras import layers
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Configuration
PROJECT_ID = "erudite-carving-472018-r5"
# DATASET_ID = "federated"
DATASET_ID = "federated_demo"

# Hospital mapping
HOSPITAL_MAPPING = {
    0: "hospital_a",
    1: "hospital_b",
    2: "hospital_c"
}

# Feature columns
NUMERICAL_FEATURES = [
    'age_at_admission', 'los_icu_hours', 'los_icu_days',
    'los_hospital_days', 'los_hospital_hours', 'n_distinct_icu_units',
    'is_mixed_icu', 'n_icu_transfers', 'n_total_transfers',
    'ed_admission_flag', 'emergency_admission_flag', 'hours_admit_to_icu',
    'early_icu_score', 'weekend_admission', 'night_admission'
]

CATEGORICAL_FEATURES = [
    'icu_type', 'first_careunit', 'admission_type',
    'admission_location', 'insurance', 'gender', 'race', 'marital_status'
]

TARGET = 'icu_mortality_label'


def load_model(input_dim: int) -> keras.Model:
    """
    Load a Multi-Layer Perceptron model for ICU mortality prediction.

    Architecture designed for:
    - Binary classification on tabular healthcare data
    - Federated learning with privacy considerations
    - Interpretability through attention mechanisms

    Args:
        input_dim: Number of input features after preprocessing

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),

        # First hidden layer with larger capacity
        layers.Dense(256, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Second hidden layer
        layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Third hidden layer
        layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Fourth hidden layer
        layers.Dense(32, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),

        # Output layer for binary classification
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile with binary crossentropy and metrics suitable for healthcare
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    return model


class DataPreprocessor:
    """Handle preprocessing of HIMAS healthcare data."""

    def __init__(self):
        self.numerical_scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False
        self.feature_dim = None

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform training data."""
        # Separate features and target
        X = df.drop(columns=[TARGET, 'stay_id', 'subject_id', 'hadm_id',
                             'icu_intime', 'icu_outtime', 'deathtime', 'death_date',
                             'assigned_hospital', 'data_split'])
        y = df[TARGET].values

        # Handle numerical features
        X_numerical = X[NUMERICAL_FEATURES].fillna(
            X[NUMERICAL_FEATURES].median())
        X_numerical_scaled = self.numerical_scaler.fit_transform(X_numerical)

        # Handle categorical features with label encoding
        X_categorical_encoded = []
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                self.label_encoders[col] = LabelEncoder()
                # Fill missing values with 'Unknown'
                col_data = X[col].fillna('Unknown').astype(str)
                encoded = self.label_encoders[col].fit_transform(col_data)
                X_categorical_encoded.append(encoded.reshape(-1, 1))

        # Combine all features
        if X_categorical_encoded:
            X_categorical_array = np.hstack(X_categorical_encoded)
            X_processed = np.hstack([X_numerical_scaled, X_categorical_array])
        else:
            X_processed = X_numerical_scaled

        self.is_fitted = True
        self.feature_dim = X_processed.shape[1]

        return X_processed, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Separate features and target
        X = df.drop(columns=[TARGET, 'stay_id', 'subject_id', 'hadm_id',
                             'icu_intime', 'icu_outtime', 'deathtime', 'death_date',
                             'assigned_hospital', 'data_split'])
        y = df[TARGET].values

        # Transform numerical features
        X_numerical = X[NUMERICAL_FEATURES].fillna(
            X[NUMERICAL_FEATURES].median())
        X_numerical_scaled = self.numerical_scaler.transform(X_numerical)

        # Transform categorical features
        X_categorical_encoded = []
        for col in CATEGORICAL_FEATURES:
            if col in X.columns and col in self.label_encoders:
                col_data = X[col].fillna('Unknown').astype(str)
                # Handle unseen labels
                le = self.label_encoders[col]
                encoded = np.array([
                    le.transform([val])[0] if val in le.classes_ else -1
                    for val in col_data
                ])
                X_categorical_encoded.append(encoded.reshape(-1, 1))

        # Combine all features
        if X_categorical_encoded:
            X_categorical_array = np.hstack(X_categorical_encoded)
            X_processed = np.hstack([X_numerical_scaled, X_categorical_array])
        else:
            X_processed = X_numerical_scaled

        return X_processed, y


# Global preprocessor cache
_preprocessor = None
_feature_dim = None


def load_data_from_bigquery(
    partition_id: int,
    num_partitions: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess data from BigQuery for a specific hospital.

    Args:
        partition_id: Hospital partition (0=hospital_a, 1=hospital_b, 2=hospital_c)
        num_partitions: Total number of partitions (should be 3 for HIMAS)

    Returns:
        Tuple of (x_train, y_train, x_val, y_val)
    """
    global _preprocessor, _feature_dim

    # Map partition_id to hospital
    hospital_name = HOSPITAL_MAPPING.get(partition_id, "hospital_a")
    table_name = f"{hospital_name}_data"

    # Initialize BigQuery client
    client = bigquery.Client(project=PROJECT_ID)

    # Query for training data
    train_query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`
    WHERE data_split = 'train'
    """

    # Query for validation data
    val_query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`
    WHERE data_split = 'validation'
    """

    print(f"Loading data for {hospital_name} (partition {partition_id})...")

    # Load data into pandas DataFrames
    df_train = client.query(train_query).to_dataframe()
    df_val = client.query(val_query).to_dataframe()

    print(f"  Training samples: {len(df_train)}")
    print(f"  Validation samples: {len(df_val)}")
    print(f"  Mortality rate (train): {df_train[TARGET].mean():.2%}")

    # Initialize preprocessor on first call
    if _preprocessor is None:
        _preprocessor = DataPreprocessor()
        x_train, y_train = _preprocessor.fit_transform(df_train)
        _feature_dim = _preprocessor.feature_dim
    else:
        x_train, y_train = _preprocessor.transform(df_train)

    # Transform validation data
    x_val, y_val = _preprocessor.transform(df_val)

    return x_train, y_train, x_val, y_val


def get_feature_dim() -> int:
    """Get the feature dimension after preprocessing."""
    global _feature_dim
    if _feature_dim is None:
        raise ValueError("Feature dimension not set. Load data first.")
    return _feature_dim
