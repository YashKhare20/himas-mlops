"""
Configuration module for HIMAS Pipeline.

Simplified configuration with essential settings only.
"""
import os
from pathlib import Path


class PipelineConfig:
    """Centralized configuration for HIMAS pipeline."""

    # ========================================================================
    # GCP CONFIGURATION
    # ========================================================================
    PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'erudite-carving-472018-r5')
    LOCATION = 'US'
    GCP_CONN_ID = 'google_cloud_default'

    # ========================================================================
    # STORAGE CONFIGURATION
    # ========================================================================
    USE_GCS = os.getenv('USE_GCS', 'false').lower() == 'true'
    GCS_BUCKET = os.getenv('GCS_REPORTS_BUCKET', 'himas-airflow-reports')

    # ========================================================================
    # PATHS
    # ========================================================================
    DAGS_DIR = Path(__file__).parent.parent
    SQL_DIR = DAGS_DIR / 'sql'
    DATA_DIR = DAGS_DIR / 'data'
    REPORTS_DIR = DATA_DIR / 'reports'

    # ========================================================================
    # EMAIL CONFIGURATION
    # ========================================================================
    ALERT_EMAIL = os.getenv('ALERT_EMAIL', 'yashkharess20@gmail.com')

    # Support multiple alert emails (comma-separated in env var)
    ALERT_EMAILS = [
            email.strip()
            for email in os.getenv('ALERT_EMAILS', ALERT_EMAIL).split(',')
    ]

    # ========================================================================
    # DATASET DEFINITIONS
    # ========================================================================
    DATASETS = [
        {
            'dataset_id': 'curated_demo',
            'description': 'HIMAS Curated Layer - Dimensional model',
            'location': LOCATION
        },
        {
            'dataset_id': 'federated_demo',
            'description': 'HIMAS Federated Layer - Hospital views',
            'location': LOCATION
        },
        {
            'dataset_id': 'verification_demo',
            'description': 'HIMAS Verification Layer - Quality checks',
            'location': LOCATION
        }
    ]