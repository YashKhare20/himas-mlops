"""
BigQuery Tools for Treatment Optimization
Queries Hospital A's historical cases and treatment protocols.

Uses data_mappings module for converting user-friendly terms to database values.

NOTE: All BigQuery results are explicitly converted to Python native types
to avoid Pydantic serialization errors with numpy types.
"""

import logging
from typing import Dict, Any, Optional, List
from google.cloud import bigquery
from datetime import datetime

from ....config import config
from ....data_mappings import (
    map_admission_type,
    map_icu_type,
    validate_early_icu_score,
    validate_query_parameters,
    get_admission_type_category,
    get_icu_type_category,
    VALID_ICU_TYPES
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# TYPE CONVERSION HELPERS
# ============================================================================

def _safe_int(value, default: int = 0) -> int:
    """Safely convert value to int, handling None and numpy types."""
    if value is None:
        return default
    try:
        if hasattr(value, '__module__') and 'numpy' in str(type(value).__module__):
            import numpy as np
            if np.isnan(value):
                return default
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float, handling None and numpy types."""
    if value is None:
        return default
    try:
        if hasattr(value, '__module__') and 'numpy' in str(type(value).__module__):
            import numpy as np
            if np.isnan(value):
                return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_str(value, default: str = "") -> str:
    """Safely convert value to string, handling None and numpy types."""
    if value is None:
        return default
    return str(value)


def _safe_list(value) -> List:
    """Safely convert value to list, handling numpy arrays."""
    if value is None:
        return []
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] if value else []


def _build_in_clause(values: List[str]) -> str:
    """Builds SQL IN clause from list of values."""
    if not values:
        return "''"
    escaped = [v.replace("'", "''") for v in values]
    return ", ".join([f"'{v}'" for v in escaped])


# ============================================================================
# QUERY FUNCTIONS
# ============================================================================

def query_similar_cases(
    age: int,
    admission_type: str,
    early_icu_score: int,
    icu_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Queries Hospital A's historical cases for similar patients.

    Args:
        age: Patient age
        admission_type: User-friendly admission type (EMERGENCY/URGENT/ELECTIVE/etc.)
        early_icu_score: 0-3 scale (will be validated)
        icu_type: Optional ICU type filter (user-friendly name)

    Returns:
        Historical case statistics with all values as Python native types
    """
    try:
        client = bigquery.Client(project=config.GOOGLE_CLOUD_PROJECT)

        # Validate and map all parameters using data_mappings
        validated = validate_query_parameters(
            admission_type=admission_type,
            icu_type=icu_type,
            early_icu_score=early_icu_score
        )

        # Get mapped values
        db_admission_types = validated['admission_types']
        db_icu_types = validated['icu_types']
        validated_score = validated['early_icu_score']

        # Build query with IN clause for multiple admission types
        admission_in_clause = _build_in_clause(db_admission_types)

        query = f"""
        SELECT 
            COUNT(*) as case_count,
            COUNT(DISTINCT subject_id) as unique_patients,
            AVG(CASE WHEN icu_mortality_label = 0 THEN 1.0 ELSE 0.0 END) as survival_rate,
            AVG(CASE WHEN icu_mortality_label = 1 THEN 1.0 ELSE 0.0 END) as mortality_rate,
            AVG(los_icu_days) as avg_los_days
        FROM `{config.get_hospital_table_id()}`
        WHERE 
            age_at_admission BETWEEN {int(age) - 5} AND {int(age) + 5}
            AND admission_type IN ({admission_in_clause})
            AND early_icu_score = {validated_score}
            AND data_split IN ('train', 'validation')
        """

        # Add ICU type filter if provided and not using all types
        if icu_type and len(db_icu_types) < len(VALID_ICU_TYPES):
            icu_in_clause = _build_in_clause(db_icu_types)
            query += f" AND icu_type IN ({icu_in_clause})"

        logger.info(
            f"Querying similar cases: age={age}±5, "
            f"admission_type={admission_type}->{db_admission_types}, "
            f"score={validated_score}"
        )

        query_job = client.query(query)
        results = list(query_job.result())

        if not results or results[0]['case_count'] == 0:
            return {
                "case_count": 0,
                "note": f"No historical cases matching criteria at {config.HOSPITAL_NAME}",
                "query_parameters": {
                    "age_range": f"{int(age) - 5}-{int(age) + 5}",
                    "admission_types_searched": db_admission_types,
                    "admission_type_category": validated['admission_type_category'],
                    "early_icu_score": validated_score,
                    "icu_types_searched": db_icu_types if icu_type else "all"
                },
                "mapping_warnings": validated['warnings']
            }

        row = results[0]

        # Extract values with explicit type conversion
        case_count = _safe_int(row['case_count'])
        unique_patients = _safe_int(row['unique_patients'])
        survival_rate = _safe_float(row['survival_rate'])
        mortality_rate = _safe_float(row['mortality_rate'])
        avg_los_days = _safe_float(row['avg_los_days'])

        return {
            "case_count": case_count,
            "unique_patients": unique_patients,
            "outcomes": {
                "survival_rate": round(survival_rate, 3),
                "mortality_rate": round(mortality_rate, 3),
                "survival_percentage": f"{round(survival_rate * 100, 1)}%"
            },
            "avg_los_days": round(avg_los_days, 1),
            "query_parameters": {
                "age_range": f"{int(age) - 5}-{int(age) + 5}",
                "admission_type_input": str(admission_type),
                "admission_types_searched": db_admission_types,
                "admission_type_category": validated['admission_type_category'],
                "early_icu_score": validated_score,
                "icu_type_input": str(icu_type) if icu_type else None,
                "icu_types_searched": db_icu_types if icu_type else "all",
                "icu_type_category": validated['icu_type_category'] if icu_type else None
            },
            "mapping_warnings": validated['warnings'],
            "data_source": f"{config.HOSPITAL_ID} historical cases",
            "query_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Similar cases query failed: {str(e)}")
        return {
            "error": str(e),
            "case_count": 0,
            "data_source": config.HOSPITAL_ID
        }


def query_treatment_protocols(
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    admission_type: Optional[str] = None,
    icu_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Queries common treatment protocols from Hospital A's medication data.

    Args:
        age_min: Minimum patient age (optional)
        age_max: Maximum patient age (optional)
        admission_type: Optional admission type filter (user-friendly)
        icu_type: Optional ICU type filter (user-friendly)

    Returns:
        Common medications and protocols with all values as Python native types
    """
    try:
        client = bigquery.Client(project=config.GOOGLE_CLOUD_PROJECT)

        # Validate parameters if provided
        validated = validate_query_parameters(
            admission_type=admission_type,
            icu_type=icu_type
        )

        query = f"""
        SELECT 
          p.drug,
          COUNT(DISTINCT p.hadm_id) AS prescription_count,
          p.drug_type,
          ARRAY_AGG(DISTINCT p.route IGNORE NULLS LIMIT 3) AS common_routes
        FROM `{config.GOOGLE_CLOUD_PROJECT}.{config.RAW_DATASET}.prescriptions` p
        INNER JOIN `{config.get_hospital_table_id()}` ha
          ON p.hadm_id = ha.hadm_id
        WHERE 1=1
        """

        # Apply age filter
        if age_min is not None and age_max is not None:
            query += f" AND ha.age_at_admission BETWEEN {int(age_min)} AND {int(age_max)}"
        elif age_min is not None:
            query += f" AND ha.age_at_admission >= {int(age_min)}"
        elif age_max is not None:
            query += f" AND ha.age_at_admission <= {int(age_max)}"

        # Apply admission type filter
        if admission_type:
            db_admission_types = validated['admission_types']
            admission_in_clause = _build_in_clause(db_admission_types)
            query += f" AND ha.admission_type IN ({admission_in_clause})"

        # Apply ICU type filter
        if icu_type and len(validated['icu_types']) < len(VALID_ICU_TYPES):
            db_icu_types = validated['icu_types']
            icu_in_clause = _build_in_clause(db_icu_types)
            query += f" AND ha.icu_type IN ({icu_in_clause})"

        query += """
        GROUP BY p.drug, p.drug_type
        HAVING prescription_count >= 2
        ORDER BY prescription_count DESC
        LIMIT 10
        """

        logger.info(
            f"Querying treatment protocols: age_min={age_min}, age_max={age_max}")

        query_job = client.query(query)
        results = list(query_job.result())

        medications = []
        for row in results:
            drug = _safe_str(row['drug'])
            drug_type = _safe_str(row['drug_type'])
            prescription_count = _safe_int(row['prescription_count'])
            common_routes = _safe_list(row['common_routes'])

            medications.append({
                "medication": drug,
                "drug_type": drug_type,
                "prescription_count": prescription_count,
                "routes": [_safe_str(r) for r in common_routes]
            })

        return {
            "common_medications": medications,
            "medication_count": len(medications),
            "query_parameters": {
                "age_min": int(age_min) if age_min is not None else None,
                "age_max": int(age_max) if age_max is not None else None,
                "admission_type_input": str(admission_type) if admission_type else None,
                "admission_types_searched": validated['admission_types'] if admission_type else None,
                "icu_type_input": str(icu_type) if icu_type else None,
                "icu_types_searched": validated['icu_types'] if icu_type else None
            },
            "mapping_warnings": validated['warnings'],
            "protocol_source": f"{config.HOSPITAL_ID} formulary",
            "query_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Treatment protocol query failed: {str(e)}")
        return {
            "error": str(e),
            "common_medications": [],
            "protocol_source": config.HOSPITAL_ID
        }


def query_outcomes_by_admission_type(
    admission_type: str,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None
) -> Dict[str, Any]:
    """
    Queries outcomes grouped by admission type.

    Args:
        admission_type: User-friendly admission type
        age_min: Minimum patient age (optional)
        age_max: Maximum patient age (optional)

    Returns:
        Outcome statistics by admission type with all values as Python native types
    """
    try:
        client = bigquery.Client(project=config.GOOGLE_CLOUD_PROJECT)

        # Map admission type
        db_admission_types, mapping_found = map_admission_type(admission_type)
        admission_in_clause = _build_in_clause(db_admission_types)

        query = f"""
        SELECT 
            admission_type,
            COUNT(DISTINCT hadm_id) as patient_count,
            AVG(CASE WHEN icu_mortality_label = 0 THEN 1.0 ELSE 0.0 END) as survival_rate,
            AVG(los_icu_days) as avg_icu_los,
            AVG(los_hospital_days) as avg_hospital_los
        FROM `{config.get_hospital_table_id()}`
        WHERE admission_type IN ({admission_in_clause})
            AND data_split IN ('train', 'validation')
        """

        if age_min is not None:
            query += f" AND age_at_admission >= {int(age_min)}"
        if age_max is not None:
            query += f" AND age_at_admission <= {int(age_max)}"

        query += " GROUP BY admission_type ORDER BY patient_count DESC"

        logger.info(
            f"Querying outcomes for admission type: {admission_type} -> {db_admission_types}")

        query_job = client.query(query)
        results = list(query_job.result())

        if not results:
            return {
                "patient_count": 0,
                "admission_type_input": str(admission_type),
                "admission_types_searched": db_admission_types,
                "note": "No patients found with this admission type"
            }

        outcomes_by_type = []
        total_patients = 0

        for row in results:
            patient_count = _safe_int(row['patient_count'])
            total_patients += patient_count

            outcomes_by_type.append({
                "admission_type": _safe_str(row['admission_type']),
                "patient_count": patient_count,
                "survival_rate": round(_safe_float(row['survival_rate']), 3),
                "avg_icu_los_days": round(_safe_float(row['avg_icu_los']), 1),
                "avg_hospital_los_days": round(_safe_float(row['avg_hospital_los']), 1)
            })

        return {
            "admission_type_input": str(admission_type),
            "admission_type_category": get_admission_type_category(db_admission_types[0]),
            "admission_types_searched": db_admission_types,
            "mapping_found": mapping_found,
            "total_patients": total_patients,
            "outcomes_by_type": outcomes_by_type,
            "data_source": f"{config.HOSPITAL_ID} historical data",
            "query_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Outcomes by admission type query failed: {str(e)}")
        return {
            "error": str(e),
            "patient_count": 0,
            "admission_type_input": str(admission_type)
        }


def query_outcomes_by_icu_type(
    icu_type: str,
    early_icu_score: Optional[int] = None
) -> Dict[str, Any]:
    """
    Queries outcomes grouped by ICU type.

    Args:
        icu_type: User-friendly ICU type
        early_icu_score: Optional severity score filter (0-3)

    Returns:
        Outcome statistics by ICU type with all values as Python native types
    """
    try:
        client = bigquery.Client(project=config.GOOGLE_CLOUD_PROJECT)

        # Map ICU type
        db_icu_types, mapping_found = map_icu_type(icu_type)
        icu_in_clause = _build_in_clause(db_icu_types)

        # Validate early ICU score if provided
        validated_score = None
        if early_icu_score is not None:
            validated_score = validate_early_icu_score(early_icu_score)

        query = f"""
        SELECT 
            icu_type,
            COUNT(DISTINCT stay_id) as stay_count,
            COUNT(DISTINCT subject_id) as patient_count,
            AVG(CASE WHEN icu_mortality_label = 0 THEN 1.0 ELSE 0.0 END) as survival_rate,
            AVG(los_icu_days) as avg_icu_los,
            AVG(early_icu_score) as avg_severity
        FROM `{config.get_hospital_table_id()}`
        WHERE icu_type IN ({icu_in_clause})
            AND data_split IN ('train', 'validation')
        """

        if validated_score is not None:
            query += f" AND early_icu_score = {validated_score}"

        query += " GROUP BY icu_type ORDER BY stay_count DESC"

        logger.info(
            f"Querying outcomes for ICU type: {icu_type} -> {db_icu_types}")

        query_job = client.query(query)
        results = list(query_job.result())

        if not results:
            return {
                "stay_count": 0,
                "icu_type_input": str(icu_type),
                "icu_types_searched": db_icu_types,
                "note": "No patients found in this ICU type"
            }

        outcomes_by_type = []
        total_stays = 0

        for row in results:
            stay_count = _safe_int(row['stay_count'])
            total_stays += stay_count

            outcomes_by_type.append({
                "icu_type": _safe_str(row['icu_type']),
                "stay_count": stay_count,
                "patient_count": _safe_int(row['patient_count']),
                "survival_rate": round(_safe_float(row['survival_rate']), 3),
                "avg_icu_los_days": round(_safe_float(row['avg_icu_los']), 1),
                "avg_severity_score": round(_safe_float(row['avg_severity']), 2)
            })

        return {
            "icu_type_input": str(icu_type),
            "icu_type_category": get_icu_type_category(db_icu_types[0]),
            "icu_types_searched": db_icu_types,
            "mapping_found": mapping_found,
            "early_icu_score_filter": validated_score,
            "total_stays": total_stays,
            "outcomes_by_type": outcomes_by_type,
            "data_source": f"{config.HOSPITAL_ID} historical data",
            "query_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Outcomes by ICU type query failed: {str(e)}")
        return {
            "error": str(e),
            "stay_count": 0,
            "icu_type_input": str(icu_type)
        }
