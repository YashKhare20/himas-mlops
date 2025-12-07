"""
Configuration for Federated Coordinator Tools

Centralized configuration for:
- GCP project settings
- BigQuery dataset names
- Privacy parameters
- Valid database values and mappings
- Hospital network configuration
"""

import os
from typing import List, Dict, Any


# ============================================================================
# JSON SERIALIZATION HELPER
# ============================================================================

def ensure_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas types to Python native types for JSON serialization.

    This function handles:
    - numpy.ndarray -> list
    - numpy.int64/int32 -> int
    - numpy.float64/float32 -> float
    - numpy.bool_ -> bool
    - numpy.str_ -> str
    - pandas Series -> list
    - Nested dicts and lists

    Args:
        obj: Any object that might contain numpy/pandas types

    Returns:
        JSON-serializable Python native types
    """
    # Handle None
    if obj is None:
        return None

    # Get type name for checking numpy types without importing numpy
    type_name = type(obj).__name__
    module_name = type(obj).__module__

    # Handle numpy types by checking module and type name
    if module_name == 'numpy' or 'numpy' in str(type(obj)):
        # numpy.ndarray
        if type_name == 'ndarray' or hasattr(obj, 'tolist'):
            return [ensure_serializable(item) for item in obj.tolist()]
        # numpy scalar types (int64, float64, etc.)
        elif 'int' in type_name.lower():
            return int(obj)
        elif 'float' in type_name.lower():
            return float(obj)
        elif 'bool' in type_name.lower():
            return bool(obj)
        elif 'str' in type_name.lower() or type_name in ('str_', 'bytes_'):
            return str(obj)
        else:
            # Fallback: try to convert to Python type
            try:
                return obj.item()
            except (AttributeError, ValueError):
                return str(obj)

    # Handle pandas Series
    if type_name == 'Series':
        return [ensure_serializable(item) for item in obj.tolist()]

    # Handle pandas DataFrame (convert to dict of lists)
    if type_name == 'DataFrame':
        return {str(k): [ensure_serializable(v) for v in vals]
                for k, vals in obj.to_dict('list').items()}

    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {str(k): ensure_serializable(v) for k, v in obj.items()}

    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [ensure_serializable(item) for item in obj]

    # Handle sets
    if isinstance(obj, set):
        return [ensure_serializable(item) for item in obj]

    # Handle standard Python types
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)

    # Handle datetime
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()

    # Fallback: convert to string
    return str(obj)

# ============================================================================
# GCP CONFIGURATION
# ============================================================================


PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "erudite-carving-472018-r5")

# ============================================================================
# BIGQUERY DATASET NAMES
# ============================================================================

# Dataset containing federated hospital data (hospital_a_data, hospital_b_data, etc.)
FEDERATED_DATASET = "federated"

# Dataset containing curated dimensional model
CURATED_DATASET = "curated"

# Dataset containing verification/statistics tables
VERIFICATION_DATASET = "verification"

# ============================================================================
# PRIVACY PARAMETERS
# ============================================================================

# Minimum group size for k-anonymity (results with fewer records are suppressed)
K_ANONYMITY_THRESHOLD = 5

# Differential privacy epsilon (lower = more privacy, more noise)
DIFFERENTIAL_PRIVACY_EPSILON = 0.1

# Age bucket size for generalization (years)
AGE_BUCKET_SIZE = 5

# Risk score bucket size for generalization
RISK_SCORE_BUCKET_SIZE = 0.1

# ============================================================================
# HOSPITAL NETWORK CONFIGURATION
# ============================================================================

# List of hospitals in the federated network
HOSPITALS = ["hospital_a", "hospital_b", "hospital_c"]

# Hospital tier priority for transfer recommendations (lower = higher priority)
HOSPITAL_TIER_PRIORITY = {
    "Tertiary Care Center": 0,
    "Community Hospital": 1,
    "Rural Hospital": 2
}

# Estimated transport times between hospitals (minutes)
TRANSPORT_TIMES = {
    ("hospital_a", "hospital_b"): 45,
    ("hospital_a", "hospital_c"): 60,
    ("hospital_b", "hospital_a"): 45,
    ("hospital_b", "hospital_c"): 75,
    ("hospital_c", "hospital_a"): 60,
    ("hospital_c", "hospital_b"): 75,
}

DEFAULT_TRANSPORT_TIME = 60

# ============================================================================
# VALID DATABASE VALUES (from federated hospital tables)
# ============================================================================

# Admission types as they appear in the database
VALID_ADMISSION_TYPES = [
    "OBSERVATION ADMIT",
    "EU OBSERVATION",
    "EW EMER.",
    "ELECTIVE",
    "SURGICAL SAME DAY ADMISSION",
    "DIRECT EMER.",
    "URGENT",
    "DIRECT OBSERVATION",
    "AMBULATORY OBSERVATION"
]

# ICU types as they appear in the database
VALID_ICU_TYPES = [
    "Cardiac ICU",
    "Medical ICU",
    "Mixed ICU (2 Units)",
    "Surgical ICU",
    "Neuro ICU",
    "Mixed ICU (3+ Units)",
    "Other ICU"
]

# Valid genders
VALID_GENDERS = ["M", "F"]

# Valid early ICU scores
VALID_EARLY_ICU_SCORES = [0, 1, 2, 3]

# First care units
VALID_FIRST_CAREUNITS = [
    "Coronary Care Unit (CCU)",
    "Medical Intensive Care Unit (MICU)",
    "Trauma SICU (TSICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)"
]

# Insurance types
VALID_INSURANCE_TYPES = ["Medicare", "Private",
                         "Other", "Medicaid", "No charge"]

# Marital statuses
VALID_MARITAL_STATUSES = ["SINGLE", "MARRIED", "DIVORCED", "WIDOWED"]

# ============================================================================
# VALUE MAPPINGS (common terms → actual DB values)
# ============================================================================

# Maps user-friendly admission type terms to actual database values
ADMISSION_TYPE_MAPPING: Dict[str, List[str]] = {
    # Emergency types
    "emergency": ["EW EMER.", "DIRECT EMER."],
    "ew emer": ["EW EMER."],
    "ew emer.": ["EW EMER."],
    "direct emer": ["DIRECT EMER."],
    "direct emer.": ["DIRECT EMER."],
    "ew emergency": ["EW EMER."],
    "direct emergency": ["DIRECT EMER."],

    # Urgent
    "urgent": ["URGENT"],

    # Elective
    "elective": ["ELECTIVE"],
    "scheduled": ["ELECTIVE"],
    "planned": ["ELECTIVE"],

    # Observation types
    "observation": ["OBSERVATION ADMIT", "EU OBSERVATION", "DIRECT OBSERVATION", "AMBULATORY OBSERVATION"],
    "observation admit": ["OBSERVATION ADMIT"],
    "eu observation": ["EU OBSERVATION"],
    "direct observation": ["DIRECT OBSERVATION"],
    "ambulatory observation": ["AMBULATORY OBSERVATION"],
    "ambulatory": ["AMBULATORY OBSERVATION"],

    # Same day surgery
    "surgical same day": ["SURGICAL SAME DAY ADMISSION"],
    "surgical same day admission": ["SURGICAL SAME DAY ADMISSION"],
    "same day": ["SURGICAL SAME DAY ADMISSION"],
    "same day surgery": ["SURGICAL SAME DAY ADMISSION"],
    "day surgery": ["SURGICAL SAME DAY ADMISSION"],
}

# Maps user-friendly ICU type terms to actual database values
ICU_TYPE_MAPPING: Dict[str, List[str]] = {
    # Cardiac
    "cardiac": ["Cardiac ICU"],
    "cardiac icu": ["Cardiac ICU"],
    "ccu": ["Cardiac ICU"],
    "cvicu": ["Cardiac ICU"],
    "coronary": ["Cardiac ICU"],

    # Medical
    "medical": ["Medical ICU"],
    "medical icu": ["Medical ICU"],
    "micu": ["Medical ICU"],

    # Surgical
    "surgical": ["Surgical ICU"],
    "surgical icu": ["Surgical ICU"],
    "sicu": ["Surgical ICU"],
    "tsicu": ["Surgical ICU"],
    "trauma": ["Surgical ICU"],

    # Neuro
    "neuro": ["Neuro ICU"],
    "neuro icu": ["Neuro ICU"],
    "neurological": ["Neuro ICU"],
    "neuro surgical": ["Neuro ICU"],

    # Mixed
    "mixed": ["Mixed ICU (2 Units)", "Mixed ICU (3+ Units)"],
    "mixed icu": ["Mixed ICU (2 Units)", "Mixed ICU (3+ Units)"],

    # Other
    "other": ["Other ICU"],
    "other icu": ["Other ICU"],
}

# ============================================================================
# PII FIELDS (for anonymization verification)
# ============================================================================

# Fields that are considered PII and must be removed/anonymized
PII_FIELDS = [
    "subject_id",
    "hadm_id",
    "name",
    "first_name",
    "last_name",
    "mrn",
    "medical_record_number",
    "dob",
    "date_of_birth",
    "ssn",
    "social_security",
    "address",
    "phone",
    "email",
    "insurance_id",
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_table_path(dataset: str, table: str) -> str:
    """
    Get full BigQuery table path.

    Args:
        dataset: Dataset name
        table: Table name

    Returns:
        Full table path: project.dataset.table
    """
    return f"{PROJECT_ID}.{dataset}.{table}"


def get_federated_table(hospital: str) -> str:
    """
    Get full path to a hospital's federated data table.

    Args:
        hospital: Hospital ID (e.g., 'hospital_a')

    Returns:
        Full table path
    """
    return get_table_path(FEDERATED_DATASET, f"{hospital}_data")


def get_transport_time(source: str, target: str) -> int:
    """
    Get estimated transport time between two hospitals.

    Args:
        source: Source hospital ID
        target: Target hospital ID

    Returns:
        Transport time in minutes
    """
    return TRANSPORT_TIMES.get((source, target), DEFAULT_TRANSPORT_TIME)
