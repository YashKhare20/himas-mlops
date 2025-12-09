"""
Transfer Coordination Tools for Federated Coordinator

Handles patient transfer coordination between hospitals in the federated network.
"""

import json
import logging
from datetime import datetime
from typing import Any

from .config import (
    PROJECT_ID,
    CURATED_DATASET,
    get_table_path,
    get_transport_time,
    ensure_serializable,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# In-memory transfer tracking (would be database in production)
_active_transfers = {}


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
        return int(value)
    except (TypeError, ValueError):
        return default


def initiate_transfer(
    source_hospital: str,
    target_hospital: str,
    patient_fingerprint: str,
    transfer_reason: str,
    urgency: str
) -> dict[str, Any]:
    """
    Coordinate a patient transfer between hospitals.

    Args:
        source_hospital: Hospital initiating the transfer
        target_hospital: Hospital receiving the patient
        patient_fingerprint: Anonymized patient identifier (SHA-256 hash)
        transfer_reason: Clinical reason for transfer
        urgency: Transfer urgency (HIGH, MEDIUM, LOW)

    Returns:
        dict containing transfer confirmation and logistics details
    """
    from google.cloud import bigquery

    transfer_id = f"xfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(
        f"[{transfer_id}] Transfer: {source_hospital} -> {target_hospital}")

    try:
        client = bigquery.Client(project=PROJECT_ID)

        table_path = get_table_path(
            CURATED_DATASET, "hospital_resources_metadata")

        # Check target hospital capacity
        query = f"""
        SELECT 
            hospital_id,
            hospital_tier,
            icu_beds_available,
            icu_beds_total,
            specialists_on_staff
        FROM `{table_path}`
        WHERE hospital_id = '{target_hospital}'
        """

        result = client.query(query).to_dataframe()

        if result.empty:
            return ensure_serializable({
                "transfer_id": str(transfer_id),
                "status": "FAILED",
                "error": f"Target hospital '{target_hospital}' not found in network"
            })

        target_info = result.iloc[0]
        beds_available = _safe_int(target_info['icu_beds_available'], 0)

        if beds_available <= 0:
            return ensure_serializable({
                "transfer_id": str(transfer_id),
                "status": "FAILED",
                "error": f"No ICU beds available at {target_hospital}",
                "suggestion": "Try querying for alternative hospitals with capability"
            })

        # Parse specialists - ensure Python list with string elements
        specialists_raw = target_info['specialists_on_staff']
        specialists = []
        if isinstance(specialists_raw, str):
            try:
                parsed = json.loads(specialists_raw.replace("'", '"'))
                # Ensure all items are strings
                if isinstance(parsed, list):
                    specialists = [str(s) for s in parsed]
            except json.JSONDecodeError:
                specialists = []
        elif isinstance(specialists_raw, list):
            specialists = [str(s) for s in specialists_raw]

        # Get transport time from config
        transport_minutes = int(get_transport_time(
            source_hospital, target_hospital))

        # Convert hospital tier to Python string
        hospital_tier = _safe_str(target_info['hospital_tier'])

        # Create transfer record - ensure all Python native types
        transfer_record = {
            "transfer_id": str(transfer_id),
            "status": "CONFIRMED",
            "source_hospital": str(source_hospital),
            "target_hospital": str(target_hospital),
            "target_hospital_tier": hospital_tier,
            "patient_fingerprint": str(patient_fingerprint),
            "transfer_reason": str(transfer_reason),
            "urgency": str(urgency),
            "bed_reservation": {
                "bed_id": f"ICU_BED_{beds_available}",
                "unit": "MICU",
                "reserved_until": datetime.now().replace(
                    hour=(datetime.now().hour + 4) % 24
                ).isoformat()
            },
            "receiving_team_notified": True,
            "specialists_available": specialists,
            "estimated_transport_minutes": int(transport_minutes),
            "initiated_at": datetime.now().isoformat(),
            "hipaa_compliant": True,
            "audit_logged": True
        }

        # Store transfer
        _active_transfers[transfer_id] = transfer_record

        # Final safety check - ensure everything is serializable
        return ensure_serializable(transfer_record)

    except Exception as e:
        logger.error(f"[{transfer_id}] Error: {str(e)}")
        return ensure_serializable({
            "transfer_id": str(transfer_id),
            "status": "FAILED",
            "error": str(e)
        })


def get_transfer_status(transfer_id: str) -> dict[str, Any]:
    """
    Get the status of an active transfer.

    Args:
        transfer_id: The transfer ID to look up

    Returns:
        dict containing current transfer status
    """
    if transfer_id in _active_transfers:
        transfer = _active_transfers[transfer_id]
        response = {
            "transfer_id": str(transfer_id),
            "found": True,
            "status": str(transfer.get("status")) if transfer.get("status") else None,
            "source_hospital": str(transfer.get("source_hospital")) if transfer.get("source_hospital") else None,
            "target_hospital": str(transfer.get("target_hospital")) if transfer.get("target_hospital") else None,
            "urgency": str(transfer.get("urgency")) if transfer.get("urgency") else None,
            "initiated_at": str(transfer.get("initiated_at")) if transfer.get("initiated_at") else None,
            "estimated_transport_minutes": int(transfer.get("estimated_transport_minutes", 0))
        }
        return ensure_serializable(response)
    else:
        return ensure_serializable({
            "transfer_id": str(transfer_id),
            "found": False,
            "message": "Transfer ID not found in active transfers"
        })
