"""
Hospital A Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# CRITICAL: Set TensorFlow environment BEFORE any TF import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Prevent TensorFlow from using multiple threads (fixes deadlocks in async)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"


class HospitalConfig:
    """Configuration for Hospital A"""

    # ========================================================================
    # HOSPITAL IDENTITY
    # ========================================================================
    HOSPITAL_ID = os.getenv('HOSPITAL_ID', 'hospital_a')
    HOSPITAL_NAME = os.getenv('HOSPITAL_NAME', 'Hospital A')

    # ========================================================================
    # A2A CONFIGURATION
    # ========================================================================
    # Each hospital agent runs on a different port
    # Federated Coordinator: 8001
    # Hospital A: 8002
    # Hospital B: 8003
    # Hospital C: 8004
    A2A_PORT = int(os.getenv('HOSPITAL_PORT', '8002'))
    A2A_HOST = os.getenv('A2A_HOST', '0.0.0.0')

    # Agent Card Metadata
    A2A_AGENT_VERSION = os.getenv('A2A_AGENT_VERSION', '1.0.0')
    A2A_PROTOCOL_VERSION = os.getenv('A2A_PROTOCOL_VERSION', '1.0')

    # ========================================================================
    # GOOGLE CLOUD CONFIGURATION
    # ========================================================================
    GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
    GOOGLE_CLOUD_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')

    # ========================================================================
    # BIGQUERY CONFIGURATION
    # ========================================================================
    BIGQUERY_DATASET = os.getenv('BIGQUERY_DATASET', 'federated')
    BIGQUERY_TABLE = os.getenv('BIGQUERY_TABLE', 'hospital_a_data')
    CURATED_DATASET = 'curated'
    RAW_DATASET = 'raw'

    # Resource Metadata Table
    RESOURCE_METADATA_TABLE = 'hospital_resources_metadata'

    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    MODEL_GCS_PATH = os.getenv('MODEL_GCS_PATH')
    MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.5-flash')

    # ========================================================================
    # FEDERATED NETWORK CONFIGURATION
    # ========================================================================
    # Federated Coordinator A2A endpoint
    FEDERATED_COORDINATOR_URL = os.getenv(
        'FEDERATED_COORDINATOR_URL',
        'http://localhost:8001'
    )
    FEDERATED_COORDINATOR_AGENT_CARD = f'{FEDERATED_COORDINATOR_URL}/.well-known/agent-card.json'

    # Peer Hospital A2A Endpoints (for direct hospital-to-hospital communication if needed)
    PEER_HOSPITALS = {
        'hospital_b': os.getenv('HOSPITAL_B_URL', 'http://localhost:8003'),
        'hospital_c': os.getenv('HOSPITAL_C_URL', 'http://localhost:8004'),
    }

    # ========================================================================
    # FEATURE CONFIGURATION (EXACT SAME as training)
    # ========================================================================

    # Target column
    TARGET_COLUMN = 'icu_mortality_label'

    # Columns to exclude from features
    EXCLUDED_COLUMNS = ['stay_id', 'subject_id', 'hadm_id', 'icu_intime', 'icu_outtime',
                        'deathtime', 'death_date', 'hospital_expire_flag',
                        'assigned_hospital', 'data_split']

    # NUMERICAL FEATURES (15 features)
    # These are scaled using StandardScaler
    NUMERICAL_FEATURES = [
        'age_at_admission',
        'hours_admit_to_icu',
        'early_icu_score',
        'los_hospital_days',
        'los_hospital_hours',
        'los_icu_hours',              # Temporal
        'los_icu_days',               # Temporal
        'n_icu_transfers',            # Temporal
        'n_total_transfers',          # Temporal
        'n_distinct_icu_units',       # Temporal
        'weekend_admission',
        'night_admission',
        'ed_admission_flag',
        'emergency_admission_flag',
        'is_mixed_icu'                # Temporal
    ]

    # CATEGORICAL FEATURES (8 features)
    # These are encoded using LabelEncoder (learns alphabetical order)
    CATEGORICAL_FEATURES = [
        'gender',
        'race',
        'marital_status',
        'insurance',
        'admission_type',
        'admission_location',
        'icu_type',
        'first_careunit'
    ]

    # Total features after preprocessing
    # Numerical: 15 (scaled to z-scores)
    # Categorical: 8 (encoded to integers via LabelEncoder)
    # Total: 23 features
    TOTAL_FEATURES = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)

    # ========================================================================
    # PRIVACY CONFIGURATION
    # ========================================================================
    DIFFERENTIAL_PRIVACY_EPSILON = float(os.getenv('DP_EPSILON', 0.1))
    K_ANONYMITY_THRESHOLD = int(os.getenv('K_ANONYMITY', 5))

    # ========================================================================
    # AUDIT LOGGING
    # ========================================================================
    AUDIT_LOG_DATASET = os.getenv('AUDIT_LOG_DATASET', 'audit_logs')
    AUDIT_LOG_TABLE = f'{HOSPITAL_ID}_audit_log'

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    @classmethod
    def get_full_table_id(cls, dataset: str, table: str) -> str:
        """Returns fully qualified BigQuery table ID."""
        return f"{cls.GOOGLE_CLOUD_PROJECT}.{dataset}.{table}"

    @classmethod
    def get_hospital_table_id(cls) -> str:
        """Returns hospital-specific data table ID."""
        return cls.get_full_table_id(cls.BIGQUERY_DATASET, cls.BIGQUERY_TABLE)

    @classmethod
    def get_resource_table_id(cls) -> str:
        """Returns hospital resources metadata table ID."""
        return cls.get_full_table_id(cls.CURATED_DATASET, cls.RESOURCE_METADATA_TABLE)


# Instantiate config
config = HospitalConfig()
