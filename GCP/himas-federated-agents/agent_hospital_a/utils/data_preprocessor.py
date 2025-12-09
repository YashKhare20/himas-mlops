"""
Data Preprocessor 

This ensures agent preprocessing EXACTLY matches training preprocessing.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
from google.cloud import bigquery

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Healthcare data preprocessing (EXACT SAME as training).
    
    **CRITICAL:** This is an exact copy of the DataPreprocessor from task.py
    to ensure agent preprocessing matches training preprocessing.
    """
    
    def __init__(self):
        """Initialize unfitted preprocessor."""
        self.numerical_scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False
        self.feature_dim = None
        logger.debug("DataPreprocessor initialized")
    
    def fit_transform(self, df: pd.DataFrame, 
                     numerical_features: list,
                     categorical_features: list,
                     target: str = 'icu_mortality_label',
                     excluded: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor on training data and return transformed features.
        
        Args:
            df: Training dataframe
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            target: Target column name
            excluded: Columns to exclude
        
        Returns:
            Tuple of (X_processed, y)
        """
        logger.info(f"Fitting preprocessor on {len(df):,} samples")
        
        if excluded is None:
            excluded = []
        
        # Separate features from target
        exclude_cols = [target] + excluded
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[target].values
        
        # Process numerical features
        logger.debug(f"Processing {len(numerical_features)} numerical features")
        X_numerical = X[numerical_features].copy()
        for col in numerical_features:
            if col in X_numerical.columns:
                X_numerical[col] = pd.to_numeric(
                    X_numerical[col], errors='coerce'
                ).astype('float64')
        X_numerical = X_numerical.fillna(X_numerical.median())
        X_numerical_scaled = self.numerical_scaler.fit_transform(X_numerical)
        
        # Process categorical features
        logger.debug(f"Processing {len(categorical_features)} categorical features")
        X_categorical_encoded = []
        for col in categorical_features:
            if col in X.columns:
                self.label_encoders[col] = LabelEncoder()
                col_data = X[col].fillna('Unknown').astype(str)
                encoded = self.label_encoders[col].fit_transform(col_data)
                X_categorical_encoded.append(encoded.reshape(-1, 1))
                logger.debug(f"  {col}: {len(self.label_encoders[col].classes_)} categories")
        
        # Combine
        if X_categorical_encoded:
            X_categorical_array = np.hstack(X_categorical_encoded)
            X_processed = np.hstack([X_numerical_scaled, X_categorical_array])
        else:
            X_processed = X_numerical_scaled
        
        self.is_fitted = True
        self.feature_dim = X_processed.shape[1]
        
        logger.info(f"Preprocessor fitted: feature_dim={self.feature_dim}")
        
        return X_processed, y
    
    def transform(self, df: pd.DataFrame,
                  numerical_features: list,
                  categorical_features: list,
                  target: str = 'icu_mortality_label',
                  excluded: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: Dataframe to transform
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            target: Target column name
            excluded: Columns to exclude
        
        Returns:
            Tuple of (X_processed, y)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if excluded is None:
            excluded = []
        
        logger.debug(f"Transforming {len(df):,} samples")
        
        # Separate features from target
        exclude_cols = [target] + excluded
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df[target].values
        
        # Transform numerical (uses TRAINING mean/std)
        X_numerical = X[numerical_features].copy()
        for col in numerical_features:
            if col in X_numerical.columns:
                X_numerical[col] = pd.to_numeric(
                    X_numerical[col], errors='coerce'
                ).astype('float64')
        X_numerical = X_numerical.fillna(X_numerical.median())
        X_numerical_scaled = self.numerical_scaler.transform(X_numerical)
        
        # Transform categorical (uses TRAINING encodings)
        X_categorical_encoded = []
        for col in categorical_features:
            if col in X.columns and col in self.label_encoders:
                col_data = X[col].fillna('Unknown').astype(str)
                le = self.label_encoders[col]
                # Handle unseen categories
                encoded = np.array([
                    le.transform([val])[0] if val in le.classes_ else -1
                    for val in col_data
                ])
                X_categorical_encoded.append(encoded.reshape(-1, 1))
        
        # Combine
        if X_categorical_encoded:
            X_categorical_array = np.hstack(X_categorical_encoded)
            X_processed = np.hstack([X_numerical_scaled, X_categorical_array])
        else:
            X_processed = X_numerical_scaled
        
        return X_processed, y


def fit_preprocessor_on_bigquery_training_data(
    project_id: str,
    dataset_id: str,
    hospital_id: str = 'hospital_a'
) -> DataPreprocessor:
    """
    Fits preprocessor on BigQuery training data (SAME as training did).
    
    This replicates the exact preprocessing from training without needing
    the saved preprocessor file.
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset (e.g., 'federated')
        hospital_id: Hospital to fit on (or 'all' for combined)
    
    Returns:
        Fitted DataPreprocessor instance
    """
    logger.info(f"Fitting preprocessor on BigQuery training data ({hospital_id})")
    
    client = bigquery.Client(project=project_id)
    
    # Load training data
    if hospital_id == 'all':
        # Fit on COMBINED training data (all hospitals)
        logger.info("Loading training data from ALL hospitals")
        all_train = []
        for hospital in ['hospital_a', 'hospital_b', 'hospital_c']:
            query = f"SELECT * FROM `{project_id}.{dataset_id}.{hospital}_data` WHERE data_split = 'train'"
            df = client.query(query).to_dataframe()
            all_train.append(df)
            logger.info(f"  {hospital}: {len(df):,} samples")
        
        df_train = pd.concat(all_train, ignore_index=True)
        logger.info(f"Combined training data: {len(df_train):,} samples")
    else:
        # Fit on single hospital's training data
        table_name = f"{hospital_id}_data"
        query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}` WHERE data_split = 'train'"
        df_train = client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df_train):,} training samples from {hospital_id}")
    
    # Define feature lists (MUST match training)
    numerical_features = [
        'age_at_admission',
        'hours_admit_to_icu',
        'early_icu_score',
        'los_hospital_days',
        'los_hospital_hours',
        'los_icu_hours',
        'los_icu_days',
        'n_icu_transfers',
        'n_total_transfers',
        'n_distinct_icu_units',
        'weekend_admission',
        'night_admission',
        'ed_admission_flag',
        'emergency_admission_flag',
        'is_mixed_icu'
    ]
    
    categorical_features = [
        'gender',
        'race',
        'marital_status',
        'insurance',
        'admission_type',
        'admission_location',
        'icu_type',
        'first_careunit'
    ]
    
    # Fit preprocessor
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(
        df_train,
        numerical_features,
        categorical_features,
        target='icu_mortality_label',
        excluded=['stay_id', 'subject_id', 'hadm_id']
    )
    
    logger.info(f"Preprocessor fitted on BigQuery data")
    logger.info(f"Feature dimension: {preprocessor.feature_dim}")
    
    return preprocessor