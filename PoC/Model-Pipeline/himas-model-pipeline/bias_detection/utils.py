"""
Bias Detection Utilities
========================

Helper functions for loading test data with demographic features preserved
for bias analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from google.cloud import bigquery
from pathlib import Path
import toml

logger = logging.getLogger(__name__)


def load_config() -> dict:
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


def create_age_groups(age: pd.Series) -> pd.Series:
    """
    Create age groups from age_at_admission.
    
    Groups:
    - <40: Age < 40
    - 40-60: 40 <= Age < 60
    - 60-75: 60 <= Age < 75
    - 75+: Age >= 75
    
    Args:
        age: Series of age values
        
    Returns:
        Series of age group labels
    """
    age_groups = pd.cut(
        age,
        bins=[0, 40, 60, 75, 150],
        labels=['<40', '40-60', '60-75', '75+'],
        right=False
    )
    return age_groups.astype(str)


def load_test_data_with_demographics(
    project_id: str,
    dataset_id: str,
    client: bigquery.Client
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load test data from BigQuery preserving demographic features.
    
    Returns both:
    1. Full dataframe with all features (for model prediction)
    2. Demographics dataframe (for bias slicing)
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        client: BigQuery client instance
        
    Returns:
        Tuple of (test_data, demographics):
        - test_data: Full test dataframe with all features
        - demographics: DataFrame with demographic features only
    """
    logger.info("Loading test data with demographic features...")
    
    # Get configuration
    hospitals = get_config_value(
        'tool.himas.data.hospital-names', 
        ['hospital_a', 'hospital_b', 'hospital_c']
    )
    test_split = get_config_value('tool.himas.data.test-split', 'test')
    
    # Demographic features to preserve
    demographic_features = ['gender', 'insurance', 'race', 'age_at_admission']
    
    all_test_data = []
    
    # Load test data from all hospitals
    for hospital in hospitals:
        query = f"""
        SELECT * 
        FROM `{project_id}.{dataset_id}.{hospital}_data`
        WHERE data_split = '{test_split}'
        """
        df = client.query(query).to_dataframe()
        all_test_data.append(df)
        logger.info(f"  {hospital}: {len(df):,} test samples")
    
    # Combine all test data
    test_data = pd.concat(all_test_data, ignore_index=True)
    logger.info(f"Total test samples: {len(test_data):,}")
    
    # Extract demographics
    demographics = test_data[demographic_features].copy()
    
    # Create age groups
    demographics['age_group'] = create_age_groups(demographics['age_at_admission'])
    demographics = demographics.drop(columns=['age_at_admission'])
    
    # Clean demographic values (handle missing/unknown)
    for col in ['gender', 'insurance', 'race']:
        demographics[col] = demographics[col].fillna('Unknown').astype(str)
    
    logger.info("Demographic distribution:")
    logger.info(f"  Gender: {demographics['gender'].value_counts().to_dict()}")
    logger.info(f"  Age groups: {demographics['age_group'].value_counts().to_dict()}")
    logger.info(f"  Insurance: {demographics['insurance'].value_counts().to_dict()}")
    logger.info(f"  Race: {demographics['race'].value_counts().to_dict()}")
    
    return test_data, demographics

