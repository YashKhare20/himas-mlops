"""
Federated Coordinator Tools Package

Tools for cross-hospital coordination with privacy guarantees.
All tool functions return JSON-serializable Python native types.
"""

from .config import (
    PROJECT_ID,
    FEDERATED_DATASET,
    CURATED_DATASET,
    VERIFICATION_DATASET,
    K_ANONYMITY_THRESHOLD,
    DIFFERENTIAL_PRIVACY_EPSILON,
    HOSPITALS,
    VALID_ADMISSION_TYPES,
    VALID_ICU_TYPES,
    ADMISSION_TYPE_MAPPING,
    ICU_TYPE_MAPPING,
    ensure_serializable,
)

from .consultation_tools import (
    query_similar_cases,
    get_admission_type_distribution,
)

from .capability_tools import (
    query_hospital_capabilities,
)

from .privacy_tools import (
    anonymize_patient_data,
    verify_k_anonymity,
)

from .statistics_tools import (
    get_network_statistics,
)

from .transfer_tools import (
    initiate_transfer,
    get_transfer_status,
)

__all__ = [
    # Config
    "PROJECT_ID",
    "FEDERATED_DATASET",
    "CURATED_DATASET",
    "VERIFICATION_DATASET",
    "K_ANONYMITY_THRESHOLD",
    "DIFFERENTIAL_PRIVACY_EPSILON",
    "HOSPITALS",
    "VALID_ADMISSION_TYPES",
    "VALID_ICU_TYPES",
    "ADMISSION_TYPE_MAPPING",
    "ICU_TYPE_MAPPING",
    "ensure_serializable",
    # Consultation
    "query_similar_cases",
    "get_admission_type_distribution",
    # Capability
    "query_hospital_capabilities",
    # Privacy
    "anonymize_patient_data",
    "verify_k_anonymity",
    # Statistics
    "get_network_statistics",
    # Transfer
    "initiate_transfer",
    "get_transfer_status",
]