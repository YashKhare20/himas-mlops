"""
Hospital Resource Check
Queries BigQuery hospital_resources_metadata table.

NOTE: All BigQuery results are explicitly converted to Python native types
to avoid Pydantic serialization errors with numpy types.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from google.cloud import bigquery
from datetime import datetime

from ....config import config
from ....data_mappings import (
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


def _safe_bool(value, default: bool = False) -> bool:
    """Safely convert value to bool, handling None and numpy types."""
    if value is None:
        return default
    try:
        if hasattr(value, 'item'):
            return bool(value.item())
        return bool(value)
    except (ValueError, TypeError):
        return default


def _safe_list(value) -> List[str]:
    """Safely convert value to list of strings, handling numpy arrays."""
    if value is None:
        return []
    try:
        if hasattr(value, 'tolist'):
            return [str(item) for item in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]
    except Exception:
        return []


# ============================================================================
# MAIN RESOURCE CHECK FUNCTION
# ============================================================================

def check_hospital_resources() -> Dict[str, Any]:
    """
    Checks hospital resources from BigQuery.

    Queries: curated.hospital_resources_metadata

    Returns:
        Complete resource availability data with all values as Python native types
    """
    try:
        client = bigquery.Client(project=config.GOOGLE_CLOUD_PROJECT)

        query = f"""
        SELECT 
            hospital_id,
            icu_beds_total,
            icu_beds_available,
            icu_beds_occupied,
            icu_occupancy_rate,
            ventilators_total,
            ventilators_available,
            ecmo_available,
            ecmo_procedures_historical,
            cardiac_surgery_available,
            cardiac_procedures_historical,
            advanced_cardiac_care_available,
            advanced_cardiac_procedures_historical,
            total_providers,
            specialists_on_staff,
            specialized_capabilities,
            hospital_tier,
            capability_summary,
            last_updated
        FROM `{config.get_resource_table_id()}`
        WHERE hospital_id = '{config.HOSPITAL_ID}'
        """

        logger.info(f"Querying resources for {config.HOSPITAL_ID}")

        query_job = client.query(query)
        results = list(query_job.result())

        if not results:
            logger.error(f"No resource data found for {config.HOSPITAL_ID}")
            return _get_fallback_resources()

        row = results[0]

        # Extract all values with explicit type conversion
        hospital_id = _safe_str(row['hospital_id'])
        hospital_tier = _safe_str(row['hospital_tier'])
        capability_summary = _safe_str(row['capability_summary'])

        # ICU beds
        icu_beds_total = _safe_int(row['icu_beds_total'])
        icu_beds_available = _safe_int(row['icu_beds_available'])
        icu_beds_occupied = _safe_int(row['icu_beds_occupied'])
        icu_occupancy_rate = _safe_float(row['icu_occupancy_rate'])

        # Ventilators
        ventilators_total = _safe_int(row['ventilators_total'])
        ventilators_available = _safe_int(row['ventilators_available'])

        # Specialized equipment
        ecmo_available = _safe_bool(row['ecmo_available'])
        ecmo_procedures = _safe_int(row['ecmo_procedures_historical'])
        cardiac_surgery_available = _safe_bool(
            row['cardiac_surgery_available'])
        cardiac_procedures = _safe_int(row['cardiac_procedures_historical'])
        advanced_cardiac_available = _safe_bool(
            row['advanced_cardiac_care_available'])
        advanced_cardiac_procedures = _safe_int(
            row['advanced_cardiac_procedures_historical'])

        # Providers
        total_providers = _safe_int(row['total_providers'])
        specialists_list = _safe_list(row['specialists_on_staff'])

        # Parse specialized capabilities
        specialized_caps = _parse_capabilities(row['specialized_capabilities'])

        # Last updated
        last_updated = row['last_updated']
        last_updated_str = last_updated.isoformat() if last_updated else None

        return {
            "hospital_id": hospital_id,
            "hospital_name": str(config.HOSPITAL_NAME),
            "hospital_tier": hospital_tier,
            "capability_summary": capability_summary,
            "query_timestamp": datetime.now().isoformat(),
            "data_freshness": last_updated_str,

            # ICU Beds
            "icu_beds": {
                "total": icu_beds_total,
                "occupied": icu_beds_occupied,
                "available": icu_beds_available,
                "occupancy_rate": icu_occupancy_rate,
                "occupancy_percentage": f"{int(icu_occupancy_rate * 100)}%",
                "status": _get_capacity_status(icu_occupancy_rate, icu_beds_available)
            },

            # Ventilators
            "ventilators": {
                "total": ventilators_total,
                "available": ventilators_available,
                "in_use": ventilators_total - ventilators_available
            },

            # Specialized Equipment & Capabilities
            "specialized_equipment": {
                "ecmo": {
                    "available": ecmo_available,
                    "historical_procedures": ecmo_procedures,
                    "status": "AVAILABLE" if ecmo_available else "NOT AVAILABLE"
                },
                "cardiac_surgery": {
                    "available": cardiac_surgery_available,
                    "historical_procedures": cardiac_procedures,
                    "status": "AVAILABLE" if cardiac_surgery_available else "NOT AVAILABLE"
                },
                "advanced_cardiac_care": {
                    "available": advanced_cardiac_available,
                    "historical_procedures": advanced_cardiac_procedures,
                    "status": "AVAILABLE" if advanced_cardiac_available else "NOT AVAILABLE",
                    "includes": "Interventional cardiology, cardiac catheterization, advanced monitoring" if advanced_cardiac_available else None
                },
                "mechanical_ventilation": specialized_caps.get('mechanical_ventilation', True),
                "dialysis": specialized_caps.get('dialysis', True)
            },

            # Specialists
            "specialists": {
                "total_providers": total_providers,
                "on_staff": specialists_list,
                "count": len(specialists_list),
                "has_cardiac_surgeon": 'cardiac_surgeon' in specialists_list,
                "has_interventional_cardiologist": 'interventional_cardiologist' in specialists_list,
                "has_infectious_disease": 'infectious_disease' in specialists_list
            },

            # Overall Status
            "overall_status": _determine_status(
                icu_occupancy_rate,
                icu_beds_available,
                specialized_caps
            ),

            # Supported ICU types (from data_mappings)
            "supported_icu_types": list(VALID_ICU_TYPES),

            # Data Source
            "data_source": "BigQuery: curated.hospital_resources_metadata (real-time)"
        }

    except Exception as e:
        logger.error(f"Resource query failed: {str(e)}")
        logger.warning("Using fallback resources")
        return _get_fallback_resources()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _parse_capabilities(capabilities_field) -> Dict[str, bool]:
    """Parses specialized_capabilities (handles STRUCT, JSON string, or dict)."""
    try:
        if capabilities_field is None:
            return _get_default_capabilities()

        if isinstance(capabilities_field, str):
            parsed = json.loads(capabilities_field)
            if 'specialized_capabilities' in parsed:
                parsed = parsed['specialized_capabilities']
            return {str(k): _safe_bool(v) for k, v in parsed.items()}

        elif isinstance(capabilities_field, dict):
            return {str(k): _safe_bool(v) for k, v in capabilities_field.items()}

        else:
            # STRUCT object from BigQuery
            result = {}
            for attr in ['ecmo', 'cardiac_surgery', 'mechanical_ventilation',
                         'dialysis', 'advanced_cardiac_care', 'infectious_disease']:
                val = getattr(capabilities_field, attr, None)
                result[attr] = _safe_bool(val)
            return result

    except Exception as e:
        logger.warning(f"Could not parse capabilities: {e}, using defaults")
        return _get_default_capabilities()


def _get_default_capabilities() -> Dict[str, bool]:
    """Returns default capability values for Hospital A."""
    return {
        'ecmo': False,
        'cardiac_surgery': True,
        'mechanical_ventilation': True,
        'dialysis': True,
        'advanced_cardiac_care': False,
        'infectious_disease': True
    }


def _get_capacity_status(occupancy_rate: float, beds_available: int) -> str:
    """Determines capacity status string."""
    if beds_available == 0:
        return "AT CAPACITY - No beds available"
    elif occupancy_rate >= 0.90:
        return "NEAR CAPACITY - Very limited availability"
    elif occupancy_rate >= 0.70:
        return "MODERATE CAPACITY"
    else:
        return "GOOD CAPACITY"


def _determine_status(
    occupancy_rate: float,
    beds_available: int,
    capabilities: Dict[str, bool]
) -> str:
    """Determines overall hospital status string."""
    capacity = _get_capacity_status(occupancy_rate, beds_available)

    limitations = []
    if not capabilities.get('advanced_cardiac_care', False):
        limitations.append("No advanced cardiac care")
    if not capabilities.get('ecmo', False):
        limitations.append("No ECMO")

    if limitations:
        return f"{capacity} | Limitations: {', '.join(limitations)}"
    return capacity


def _get_fallback_resources() -> Dict[str, Any]:
    """Conservative fallback if BigQuery fails - all values are Python native types."""
    return {
        "hospital_id": str(config.HOSPITAL_ID),
        "hospital_name": str(config.HOSPITAL_NAME),
        "error": "BigQuery query failed - using fallback",
        "query_timestamp": datetime.now().isoformat(),
        "icu_beds": {
            "total": 8,
            "available": 1,
            "occupied": 7,
            "occupancy_rate": 0.88,
            "occupancy_percentage": "88%",
            "status": "NEAR CAPACITY (fallback data)"
        },
        "ventilators": {
            "total": 8,
            "available": 2,
            "in_use": 6
        },
        "specialized_equipment": {
            "ecmo": {
                "available": False,
                "historical_procedures": 0,
                "status": "NOT AVAILABLE"
            },
            "advanced_cardiac_care": {
                "available": False,
                "historical_procedures": 0,
                "status": "NOT AVAILABLE"
            },
            "cardiac_surgery": {
                "available": True,
                "historical_procedures": 0,
                "status": "AVAILABLE"
            },
            "mechanical_ventilation": True,
            "dialysis": True
        },
        "specialists": {
            "total_providers": 10,
            "on_staff": ["pulmonologist", "cardiologist", "intensivist"],
            "count": 3,
            "has_cardiac_surgeon": False,
            "has_interventional_cardiologist": False,
            "has_infectious_disease": True
        },
        "overall_status": "UNCERTAIN - Resource data unavailable",
        "supported_icu_types": list(VALID_ICU_TYPES),
        "data_source": "Fallback configuration"
    }


def check_specific_capability(capability_name: str) -> Dict[str, Any]:
    """
    Checks if a specific capability is available at this hospital.

    Args:
        capability_name: Name of capability to check (e.g., 'ecmo', 'cardiac_surgery')

    Returns:
        Capability status with all values as Python native types
    """
    try:
        resources = check_hospital_resources()

        if "error" in resources and "fallback" in resources.get("error", "").lower():
            return {
                "capability": str(capability_name),
                "available": False,
                "status": "UNKNOWN - using fallback data",
                "hospital_id": str(config.HOSPITAL_ID)
            }

        # Check in specialized equipment
        specialized = resources.get("specialized_equipment", {})

        if capability_name in specialized:
            cap_info = specialized[capability_name]
            if isinstance(cap_info, dict):
                return {
                    "capability": str(capability_name),
                    "available": bool(cap_info.get("available", False)),
                    "status": str(cap_info.get("status", "UNKNOWN")),
                    "historical_procedures": int(cap_info.get("historical_procedures", 0)),
                    "hospital_id": str(config.HOSPITAL_ID),
                    "hospital_tier": str(resources.get("hospital_tier", "Unknown"))
                }
            else:
                return {
                    "capability": str(capability_name),
                    "available": bool(cap_info),
                    "status": "AVAILABLE" if cap_info else "NOT AVAILABLE",
                    "hospital_id": str(config.HOSPITAL_ID)
                }

        # Check specialists
        specialist_mapping = {
            "cardiac_surgeon": "has_cardiac_surgeon",
            "interventional_cardiologist": "has_interventional_cardiologist",
            "infectious_disease": "has_infectious_disease"
        }

        if capability_name in specialist_mapping:
            specialists = resources.get("specialists", {})
            has_specialist = specialists.get(
                specialist_mapping[capability_name], False)
            return {
                "capability": str(capability_name),
                "available": bool(has_specialist),
                "status": "ON STAFF" if has_specialist else "NOT ON STAFF",
                "hospital_id": str(config.HOSPITAL_ID)
            }

        return {
            "capability": str(capability_name),
            "available": False,
            "status": "CAPABILITY NOT FOUND",
            "hospital_id": str(config.HOSPITAL_ID),
            "note": f"Unknown capability: {capability_name}"
        }

    except Exception as e:
        logger.error(f"Capability check failed: {str(e)}")
        return {
            "capability": str(capability_name),
            "available": False,
            "status": "ERROR",
            "error": str(e),
            "hospital_id": str(config.HOSPITAL_ID)
        }


def get_icu_capacity_by_type() -> Dict[str, Any]:
    """
    Gets ICU capacity breakdown by type (using valid ICU types from data_mappings).

    Returns:
        ICU capacity by type with all values as Python native types
    """
    try:
        client = bigquery.Client(project=config.GOOGLE_CLOUD_PROJECT)

        query = f"""
        SELECT 
            icu_type,
            COUNT(*) as current_patients,
            AVG(los_icu_days) as avg_los_days
        FROM `{config.get_hospital_table_id()}`
        WHERE data_split = 'train'
        GROUP BY icu_type
        ORDER BY current_patients DESC
        """

        query_job = client.query(query)
        results = list(query_job.result())

        capacity_by_type = []
        for row in results:
            icu_type = _safe_str(row['icu_type'])
            capacity_by_type.append({
                "icu_type": icu_type,
                "icu_type_category": get_icu_type_category(icu_type),
                "historical_patients": _safe_int(row['current_patients']),
                "avg_los_days": round(_safe_float(row['avg_los_days']), 1)
            })

        return {
            "hospital_id": str(config.HOSPITAL_ID),
            "capacity_by_icu_type": capacity_by_type,
            "valid_icu_types": list(VALID_ICU_TYPES),
            "query_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"ICU capacity query failed: {str(e)}")
        return {
            "error": str(e),
            "hospital_id": str(config.HOSPITAL_ID),
            "valid_icu_types": list(VALID_ICU_TYPES)
        }
