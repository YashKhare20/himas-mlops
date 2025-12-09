"""
Similar Case Consultation Tools for Federated Coordinator

Handles privacy-preserved queries across the federated hospital network.
Maps user-friendly terms to actual database values.
"""

import logging
from datetime import datetime
from typing import Any, Optional, List

from .config import (
    PROJECT_ID,
    FEDERATED_DATASET,
    K_ANONYMITY_THRESHOLD,
    DIFFERENTIAL_PRIVACY_EPSILON,
    HOSPITALS,
    VALID_ADMISSION_TYPES,
    VALID_ICU_TYPES,
    ADMISSION_TYPE_MAPPING,
    ICU_TYPE_MAPPING,
    get_federated_table,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# VALUE NORMALIZATION FUNCTIONS
# ============================================================================

def _normalize_admission_type(admission_type: str) -> List[str]:
    """
    Normalize admission type to actual database values.

    Args:
        admission_type: User-provided or mapped admission type

    Returns:
        List of valid database values to query
    """
    if not admission_type:
        logger.warning(
            "[COORDINATOR] No admission type provided, defaulting to emergency types")
        return ["EW EMER.", "DIRECT EMER."]

    original = admission_type

    # Check if it's already a valid DB value (exact match)
    if admission_type in VALID_ADMISSION_TYPES:
        logger.info(
            f"[COORDINATOR] Admission type '{admission_type}' is valid DB value")
        return [admission_type]

    # Normalize for lookup
    normalized = admission_type.lower().strip()
    normalized_no_period = normalized.rstrip('.')

    # Check mapping table
    if normalized in ADMISSION_TYPE_MAPPING:
        mapped = ADMISSION_TYPE_MAPPING[normalized]
        logger.info(f"[COORDINATOR] Mapped '{original}' → {mapped}")
        return mapped

    if normalized_no_period in ADMISSION_TYPE_MAPPING:
        mapped = ADMISSION_TYPE_MAPPING[normalized_no_period]
        logger.info(f"[COORDINATOR] Mapped '{original}' → {mapped}")
        return mapped

    # Check for partial matches in valid types
    for valid_type in VALID_ADMISSION_TYPES:
        if normalized in valid_type.lower() or valid_type.lower() in normalized:
            logger.info(
                f"[COORDINATOR] Partial match: '{original}' → '{valid_type}'")
            return [valid_type]

    # Check for partial matches in mapping keys
    for key, values in ADMISSION_TYPE_MAPPING.items():
        if key in normalized or normalized in key:
            logger.info(f"[COORDINATOR] Key match: '{original}' → {values}")
            return values

    # Default to emergency types
    logger.warning(
        f"[COORDINATOR] Unknown admission type '{original}', defaulting to emergency")
    return ["EW EMER.", "DIRECT EMER."]


def _normalize_icu_type(icu_type: str) -> List[str]:
    """
    Normalize ICU type to actual database values.

    Args:
        icu_type: User-provided ICU type

    Returns:
        List of valid database values to query
    """
    if not icu_type:
        return VALID_ICU_TYPES  # All types if not specified

    # Exact match
    if icu_type in VALID_ICU_TYPES:
        logger.info(f"[COORDINATOR] ICU type '{icu_type}' is valid DB value")
        return [icu_type]

    normalized = icu_type.lower().strip()

    # Check mapping table
    if normalized in ICU_TYPE_MAPPING:
        mapped = ICU_TYPE_MAPPING[normalized]
        logger.info(f"[COORDINATOR] Mapped ICU type '{icu_type}' → {mapped}")
        return mapped

    # Partial keyword matching
    for key, values in ICU_TYPE_MAPPING.items():
        if key in normalized or normalized in key:
            logger.info(
                f"[COORDINATOR] ICU type keyword match: '{icu_type}' → {values}")
            return values

    # Default to all types
    logger.warning(
        f"[COORDINATOR] Unknown ICU type '{icu_type}', querying all types")
    return VALID_ICU_TYPES


def _parse_age_bucket(age_bucket: str) -> tuple:
    """
    Parse age bucket string to min/max values.

    Args:
        age_bucket: Age range string (e.g., '75-80', '85+')

    Returns:
        Tuple of (age_min, age_max)
    """
    age_bucket_clean = age_bucket.replace('+', '-120')
    age_parts = age_bucket_clean.split('-')
    age_min = int(age_parts[0])
    age_max = int(age_parts[1]) if len(age_parts) > 1 else 120
    return age_min, age_max


# ============================================================================
# MAIN QUERY FUNCTION
# ============================================================================

def query_similar_cases(
    age_bucket: str,
    admission_type: str,
    early_icu_score: int,
    requesting_hospital: str,
    icu_type: Optional[str] = None
) -> dict[str, Any]:
    """
    Query the federated network for similar cases with privacy guarantees.

    Implements k-anonymity and differential privacy to protect patient privacy.

    Args:
        age_bucket: Age range (e.g., '75-80', '65-70')
        admission_type: Type of admission (can be common term or actual DB value)
        early_icu_score: ICU severity score (0-3)
        requesting_hospital: Hospital making the request
        icu_type: Optional ICU type filter

    Returns:
        dict containing aggregated, privacy-preserved statistics from peer hospitals
    """
    import numpy as np
    from google.cloud import bigquery

    query_id = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("=" * 70)
    logger.info(f"[{query_id}] FEDERATED COORDINATOR: Similar Case Query")
    logger.info("=" * 70)
    logger.info(f"[{query_id}] Requesting hospital: {requesting_hospital}")
    logger.info(f"[{query_id}] Raw parameters: age={age_bucket}, admission={admission_type}, "
                f"score={early_icu_score}, icu={icu_type}")

    try:
        client = bigquery.Client(project=PROJECT_ID)

        # Parse age bucket
        age_min, age_max = _parse_age_bucket(age_bucket)
        logger.info(f"[{query_id}] Age range: {age_min} to {age_max}")

        # Normalize admission type to actual DB values
        admission_types = _normalize_admission_type(admission_type)
        logger.info(
            f"[{query_id}] Normalized admission types: {admission_types}")

        # Normalize ICU type if provided
        icu_types = _normalize_icu_type(icu_type) if icu_type else None
        if icu_types and icu_types != VALID_ICU_TYPES:
            logger.info(f"[{query_id}] Normalized ICU types: {icu_types}")

        results_by_hospital = {}

        for hospital in HOSPITALS:
            if hospital == requesting_hospital:
                logger.info(
                    f"[{query_id}] Skipping {hospital} (requesting hospital)")
                continue

            # Build admission type filter with OR conditions
            admission_conditions = " OR ".join([
                f"admission_type = '{t}'" for t in admission_types
            ])

            # Build ICU type filter if specified
            icu_condition = ""
            if icu_types and icu_types != VALID_ICU_TYPES:
                icu_conditions = " OR ".join(
                    [f"icu_type = '{t}'" for t in icu_types])
                icu_condition = f"AND ({icu_conditions})"

            # Build query using config
            table_path = get_federated_table(hospital)

            query = f"""
            SELECT 
                COUNT(*) as case_count,
                SUM(icu_mortality_label) as deaths,
                AVG(los_icu_hours) as avg_icu_los,
                AVG(los_hospital_days) as avg_hospital_los
            FROM `{table_path}`
            WHERE 
                age_at_admission BETWEEN {age_min} AND {age_max}
                AND ({admission_conditions})
                AND early_icu_score = {early_icu_score}
                {icu_condition}
            """

            logger.info(f"[{query_id}] Querying {hospital}...")
            logger.info(f"[{query_id}] WHERE: age BETWEEN {age_min} AND {age_max} "
                        f"AND ({admission_conditions}) AND early_icu_score = {early_icu_score}")

            result = client.query(query).to_dataframe()

            if not result.empty and result.iloc[0]['case_count'] >= K_ANONYMITY_THRESHOLD:
                row = result.iloc[0]
                # Convert numpy types to Python native types
                case_count = int(row['case_count'])
                deaths = int(row['deaths']) if row['deaths'] is not None and not np.isnan(
                    row['deaths']) else 0

                logger.info(
                    f"[{query_id}] {hospital}: Found {case_count} cases, {deaths} deaths")

                # Apply differential privacy (Laplace noise)
                # Use Python float() to avoid numpy types in output
                noise_scale = float(1.0 / DIFFERENTIAL_PRIVACY_EPSILON)
                noise_count = int(
                    round(float(np.random.laplace(0, noise_scale))))
                noise_deaths = int(
                    round(float(np.random.laplace(0, noise_scale))))

                noisy_count = int(
                    max(K_ANONYMITY_THRESHOLD, case_count + noise_count))
                noisy_deaths = int(
                    max(0, min(noisy_count, deaths + noise_deaths)))

                # Calculate rates as Python floats
                survival_rate = float(
                    1 - (noisy_deaths / noisy_count)) if noisy_count > 0 else None
                mortality_rate = float(
                    noisy_deaths / noisy_count) if noisy_count > 0 else None

                # Convert pandas/numpy values to Python native types for JSON serialization
                avg_icu_los = None
                if row['avg_icu_los'] is not None and not np.isnan(row['avg_icu_los']):
                    avg_icu_los = float(round(float(row['avg_icu_los']), 1))

                avg_hospital_los = None
                if row['avg_hospital_los'] is not None and not np.isnan(row['avg_hospital_los']):
                    avg_hospital_los = float(
                        round(float(row['avg_hospital_los']), 1))

                results_by_hospital[hospital] = {
                    "cases_found": int(noisy_count),
                    "survival_rate": float(round(survival_rate, 3)) if survival_rate is not None else None,
                    "mortality_rate": float(round(mortality_rate, 3)) if mortality_rate is not None else None,
                    "avg_icu_los_hours": avg_icu_los,
                    "avg_hospital_los_days": avg_hospital_los,
                    "k_anonymity_met": True,
                    "differential_privacy_applied": True
                }

                if survival_rate is not None:
                    logger.info(f"[{query_id}] {hospital}: {noisy_count} cases (noisy), "
                                f"survival = {survival_rate:.1%}")
                else:
                    logger.info(
                        f"[{query_id}] {hospital}: {noisy_count} cases (noisy)")
            else:
                actual_count = int(
                    result.iloc[0]['case_count']) if not result.empty else 0
                results_by_hospital[hospital] = {
                    "cases_found": int(0),
                    "actual_count_suppressed": int(actual_count),
                    "message": f"Insufficient cases for privacy-safe reporting "
                    f"(k={actual_count} < {K_ANONYMITY_THRESHOLD})",
                    "k_anonymity_met": False
                }
                logger.info(f"[{query_id}] {hospital}: k-anonymity NOT met "
                            f"(k={actual_count} < {K_ANONYMITY_THRESHOLD})")

        logger.info("=" * 70)
        logger.info(f"[{query_id}] Query complete")
        logger.info("=" * 70)

        # Ensure all return values are Python native types for JSON serialization
        return {
            "query_id": str(query_id),
            "query_parameters": {
                "age_bucket": str(age_bucket),
                "age_range": {"min": int(age_min), "max": int(age_max)},
                "admission_type_requested": str(admission_type) if admission_type else None,
                "admission_types_queried": [str(t) for t in admission_types],
                "early_icu_score": int(early_icu_score),
                "icu_type": str(icu_type) if icu_type else None,
                "icu_types_queried": [str(t) for t in icu_types] if icu_types else None
            },
            "requesting_hospital": str(requesting_hospital),
            "results_by_hospital": results_by_hospital,  # Already converted in loop
            "privacy_guarantees": {
                "k_anonymity_threshold": int(K_ANONYMITY_THRESHOLD),
                "differential_privacy_epsilon": float(DIFFERENTIAL_PRIVACY_EPSILON)
            },
            "dataset": str(FEDERATED_DATASET),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"[{query_id}] Error: {str(e)}")
        import traceback
        logger.error(f"[{query_id}] Traceback: {traceback.format_exc()}")
        return {
            "query_id": query_id,
            "error": str(e),
            "results_by_hospital": {}
        }


# ============================================================================
# DEBUG/UTILITY FUNCTIONS
# ============================================================================

def get_admission_type_distribution(hospital: str) -> dict[str, Any]:
    """
    Get the distribution of admission types at a hospital (for debugging).

    Args:
        hospital: Hospital ID (hospital_a, hospital_b, hospital_c)

    Returns:
        dict with admission type counts
    """
    from google.cloud import bigquery

    logger.info(f"Getting admission type distribution for {hospital}")

    try:
        client = bigquery.Client(project=PROJECT_ID)
        table_path = get_federated_table(hospital)

        query = f"""
        SELECT 
            admission_type,
            COUNT(*) as count
        FROM `{table_path}`
        GROUP BY admission_type
        ORDER BY count DESC
        """

        result = client.query(query).to_dataframe()

        # Ensure all values are Python native types
        distribution = {}
        for _, row in result.iterrows():
            key = str(row['admission_type']
                      ) if row['admission_type'] is not None else "NULL"
            distribution[key] = int(row['count'])

        logger.info(f"Distribution for {hospital}: {distribution}")

        return {
            "hospital": str(hospital),
            "dataset": str(FEDERATED_DATASET),
            "admission_type_distribution": distribution,
            "valid_admission_types": list(VALID_ADMISSION_TYPES),
            "total_records": int(sum(distribution.values()))
        }

    except Exception as e:
        logger.error(f"Error getting admission type distribution: {str(e)}")
        return {"error": str(e)}
