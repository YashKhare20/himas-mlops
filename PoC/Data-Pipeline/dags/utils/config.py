"""
Configuration module for HIMAS Pipeline.

Centralized configuration with table definitions for proper execution order.
"""
import os
from pathlib import Path


class PipelineConfig:
    """Centralized configuration for HIMAS pipeline."""

    # ========================================================================
    # GCP CONFIGURATION
    # ========================================================================
    PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'erudite-carving-472018-r5')
    LOCATION = 'US'
    GCP_CONN_ID = 'google_cloud_default'

    # ========================================================================
    # STORAGE CONFIGURATION
    # ========================================================================
    USE_GCS = os.getenv('USE_GCS', 'false').lower() == 'true'
    GCS_BUCKET = os.getenv('GCS_BUCKET', 'himas-airflow-data')

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

    # ========================================================================
    # TABLE DEFINITIONS (EXECUTION ORDER)
    # ========================================================================

    # CURATED LAYER - Dimensional Model
    # Order matters! Follow dimensional modeling best practices
    CURATED_TABLES = [
        "patient_split_assignment",  # 1. Base: Patient-to-hospital assignments
        "dim_patient",                # 2. Dimension: Patient demographics
        "fact_hospital_admission",    # 3. Fact: Hospital admission events
        # 4. Fact: ICU stay details (depends on admissions)
        "fact_icu_stay",
        # 5. Fact: Patient transfers (depends on admissions)
        "fact_transfers",
        # 6. Aggregate: Clinical features (depends on all above)
        "clinical_features"
    ]

    # FEDERATED LAYER - Hospital Partitions
    # Can run in parallel (no dependencies between hospitals)
    FEDERATED_TABLES = [
        "hospital_a_data",  # Hospital A: 40% of patients
        "hospital_b_data",  # Hospital B: 35% of patients
        "hospital_c_data"   # Hospital C: 25% of patients
    ]

    # VERIFICATION LAYER - Quality Checks
    # Can run in parallel (independent checks)
    VERIFICATION_TABLES = [
        "data_leakage_check",  # Check for patient overlap between splits/hospitals
        "dataset_statistics"   # Generate comprehensive dataset statistics
    ]

    # ========================================================================
    # TABLE METADATA (OPTIONAL - FOR DOCUMENTATION)
    # ========================================================================

    TABLE_DESCRIPTIONS = {
        # Curated Layer
        "patient_split_assignment": "Assigns patients to train/val/test splits and hospitals",
        "dim_patient": "Patient demographic dimension table",
        "fact_hospital_admission": "Hospital admission fact table",
        "fact_icu_stay": "ICU stay fact table with mortality indicators",
        "fact_transfers": "Patient transfer events fact table",
        "clinical_features": "Aggregated clinical features for ML modeling",

        # Federated Layer
        "hospital_a_data": "Hospital A federated dataset (40% patients)",
        "hospital_b_data": "Hospital B federated dataset (35% patients)",
        "hospital_c_data": "Hospital C federated dataset (25% patients)",

        # Verification Layer
        "data_leakage_check": "Validates no patient leakage between splits/hospitals",
        "dataset_statistics": "Comprehensive statistics for all datasets"
    }

    # ========================================================================
    # LAYER CONFIGURATION
    # ========================================================================

    LAYERS = {
        "curated": {
            "tables": CURATED_TABLES,
            "sql_dir": "curated_layer",
            "dataset_id": "curated_demo",
            "sequential": True,  # Must run in order
            "description": "Dimensional model with patient splits"
        },
        "federated": {
            "tables": FEDERATED_TABLES,
            "sql_dir": "federated_layer",
            "dataset_id": "federated_demo",
            "sequential": False,  # Can run in parallel
            "description": "Hospital-specific partitioned data"
        },
        "verification": {
            "tables": VERIFICATION_TABLES,
            "sql_dir": "verification_layer",
            "dataset_id": "verification_demo",
            "sequential": False,  # Can run in parallel
            "description": "Data quality validation checks"
        }
    }

    # ========================================================================
    # VALIDATION CONFIGURATION
    # ========================================================================

    # Tables that require validation
    VALIDATE_TABLES = {
        "curated": CURATED_TABLES,
        "federated": FEDERATED_TABLES,
        "verification": VERIFICATION_TABLES
    }

    # Tables that require statistics
    STATISTICS_TABLES = [
        ("curated_demo", "dim_patient"),
        ("curated_demo", "fact_icu_stay"),
        ("federated_demo", "hospital_a_data"),
        ("federated_demo", "hospital_b_data"),
        ("federated_demo", "hospital_c_data"),
        ("verification_demo", "dataset_statistics")
    ]

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    @classmethod
    def get_layer_config(cls, layer_name: str) -> dict:
        """
        Get configuration for a specific layer.

        Args:
            layer_name: Name of the layer (curated, federated, verification)

        Returns:
            Dictionary with layer configuration
        """
        return cls.LAYERS.get(layer_name, {})

    @classmethod
    def get_table_full_id(cls, dataset_id: str, table_name: str) -> str:
        """
        Get full BigQuery table ID.

        Args:
            dataset_id: Dataset ID
            table_name: Table name

        Returns:
            Full table ID in format: project.dataset.table
        """
        return f"{cls.PROJECT_ID}.{dataset_id}.{table_name}"

    @classmethod
    def get_all_tables(cls) -> list:
        """
        Get list of all tables across all layers.

        Returns:
            List of tuples (dataset_id, table_name)
        """
        all_tables = []
        for layer in cls.LAYERS.values():
            dataset_id = layer['dataset_id']
            for table in layer['tables']:
                all_tables.append((dataset_id, table))
        return all_tables
