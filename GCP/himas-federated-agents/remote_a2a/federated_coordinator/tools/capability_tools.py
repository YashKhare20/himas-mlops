"""
Capability Query Tools for Federated Coordinator

Queries hospitals for specific capabilities and recommends transfer targets.
"""

import json
import logging
from datetime import datetime
from typing import Any

from .config import (
    PROJECT_ID,
    CURATED_DATASET,
    HOSPITAL_TIER_PRIORITY,
    get_table_path,
    ensure_serializable,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _safe_str(value, default=None):
    """Safely convert a value to Python str, handling numpy and None."""
    if value is None:
        return default
    try:
        return str(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    """Safely convert a value to Python int, handling numpy and None."""
    if value is None:
        return default
    try:
        # Handle numpy types
        type_name = type(value).__name__
        if 'int' in type_name.lower() or 'float' in type_name.lower():
            return int(value)
        return int(value)
    except (TypeError, ValueError):
        return default


def query_hospital_capabilities(
    required_capability: str,
    requesting_hospital: str
) -> dict[str, Any]:
    """
    Query all hospitals in the federated network for a specific capability.

    Args:
        required_capability: The capability needed (e.g., 'advanced_cardiac_care', 
                           'ecmo', 'infectious_disease', 'cardiac_surgery')
        requesting_hospital: The hospital making the request (for audit logging)

    Returns:
        dict containing hospitals with/without capability and transfer recommendation
    """
    from google.cloud import bigquery

    query_id = f"cap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(
        f"[{query_id}] Capability query: {required_capability} from {requesting_hospital}")

    try:
        client = bigquery.Client(project=PROJECT_ID)

        table_path = get_table_path(
            CURATED_DATASET, "hospital_resources_metadata")

        query = f"""
        SELECT 
            hospital_id,
            hospital_tier,
            icu_beds_available,
            specialized_capabilities,
            capability_summary
        FROM `{table_path}`
        """

        results = client.query(query).to_dataframe()

        hospitals_with = []
        hospitals_without = []

        for _, row in results.iterrows():
            # Convert numpy strings to Python strings
            hospital_id = _safe_str(row['hospital_id'])

            # Skip requesting hospital
            if hospital_id == requesting_hospital:
                continue

            capabilities = row['specialized_capabilities']
            has_capability = False

            if isinstance(capabilities, dict):
                has_capability = bool(
                    capabilities.get(required_capability, False))
            elif isinstance(capabilities, str):
                try:
                    cap_dict = json.loads(capabilities)
                    has_capability = bool(
                        cap_dict.get(required_capability, False))
                except json.JSONDecodeError:
                    pass

            # Ensure all values are Python native types
            hospital_info = {
                "hospital_id": str(hospital_id) if hospital_id else None,
                "hospital_tier": _safe_str(row['hospital_tier']),
                "icu_beds_available": _safe_int(row['icu_beds_available'], 0),
                "capability_summary": _safe_str(row['capability_summary'])
            }

            if has_capability:
                hospitals_with.append(hospital_info)
            else:
                hospitals_without.append(hospital_info)

        # Determine best transfer target (prioritize tier, then beds)
        recommended_target = None
        if hospitals_with:
            hospitals_with.sort(key=lambda x: (
                HOSPITAL_TIER_PRIORITY.get(x['hospital_tier'], 99),
                -x['icu_beds_available']
            ))
            recommended_target = str(hospitals_with[0]['hospital_id'])

        response = {
            "query_id": str(query_id),
            "required_capability": str(required_capability),
            "requesting_hospital": str(requesting_hospital),
            "hospitals_with_capability": hospitals_with,
            "hospitals_without_capability": hospitals_without,
            "recommended_transfer_target": recommended_target,
            "timestamp": datetime.now().isoformat()
        }

        # Final safety check - ensure everything is serializable
        return ensure_serializable(response)

    except Exception as e:
        logger.error(f"[{query_id}] Error: {str(e)}")
        return ensure_serializable({
            "query_id": str(query_id),
            "error": str(e),
            "hospitals_with_capability": [],
            "recommended_transfer_target": None
        })
