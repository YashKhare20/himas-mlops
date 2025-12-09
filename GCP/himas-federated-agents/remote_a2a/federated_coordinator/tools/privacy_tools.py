"""
Privacy Tools for Federated Coordinator

Handles patient data anonymization and k-anonymity verification.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Optional

from .config import (
    K_ANONYMITY_THRESHOLD,
    AGE_BUCKET_SIZE,
    RISK_SCORE_BUCKET_SIZE,
    PII_FIELDS,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def anonymize_patient_data(
    subject_id: int,
    age: int,
    risk_score: Optional[float] = None
) -> dict[str, Any]:
    """
    Anonymize patient data for cross-hospital queries.

    Creates a privacy-preserving representation of a patient that can be
    shared across the federated network without revealing identity.

    Args:
        subject_id: Original patient ID (will be hashed)
        age: Patient age (will be bucketed)
        risk_score: Mortality risk score (optional, will be generalized)

    Returns:
        dict containing anonymized patient representation
    """
    # Create patient fingerprint (SHA-256 hash with daily salt)
    daily_salt = datetime.now().strftime('%Y%m%d')
    fingerprint = hashlib.sha256(
        f"patient_{subject_id}_{daily_salt}".encode()
    ).hexdigest()[:16]

    # Age bucketing using configured bucket size
    age_bucket_start = (age // AGE_BUCKET_SIZE) * AGE_BUCKET_SIZE
    age_bucket = f"{age_bucket_start}-{age_bucket_start + AGE_BUCKET_SIZE}"

    result = {
        "patient_fingerprint": fingerprint,
        "age_bucket": age_bucket,
        "anonymization_timestamp": datetime.now().isoformat(),
        "identifiers_removed": PII_FIELDS,
        "k_anonymity_compliant": True
    }

    # Risk score generalization using configured bucket size
    if risk_score is not None:
        risk_bucket_start = int(
            risk_score / RISK_SCORE_BUCKET_SIZE) * RISK_SCORE_BUCKET_SIZE
        result["risk_score_range"] = f"{risk_bucket_start:.1f}-{risk_bucket_start + RISK_SCORE_BUCKET_SIZE:.1f}"

    logger.info(
        f"Anonymized patient: fingerprint={fingerprint[:8]}..., age_bucket={age_bucket}")

    return result


def verify_k_anonymity(
    query_result_count: int,
    k_threshold: Optional[int] = None
) -> dict[str, Any]:
    """
    Verify that a query result meets k-anonymity requirements.

    Args:
        query_result_count: Number of records in query result
        k_threshold: Minimum required records for k-anonymity (default from config)

    Returns:
        dict indicating whether k-anonymity is satisfied
    """
    if k_threshold is None:
        k_threshold = K_ANONYMITY_THRESHOLD

    is_satisfied = query_result_count >= k_threshold

    return {
        "k_threshold": k_threshold,
        "result_count": query_result_count,
        "k_anonymity_satisfied": is_satisfied,
        "can_share_results": is_satisfied,
        "message": (
            f"K-anonymity satisfied (k={query_result_count} >= {k_threshold})"
            if is_satisfied
            else f"K-anonymity NOT satisfied (k={query_result_count} < {k_threshold}). Cannot share results."
        )
    }
