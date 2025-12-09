"""
Resource Allocation Tools
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import from treatment_optimization tools
from ...treatment_optimization.tools.resource_check import check_hospital_resources

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_bed_availability() -> Dict[str, Any]:
    """
    Checks ICU bed availability (uses shared resource_check function).

    Returns:
        Bed availability data
    """
    # Delegate to shared function
    resources = check_hospital_resources()

    # Return just bed-specific data
    return {
        "hospital_id": resources["hospital_id"],
        "icu_beds": resources["icu_beds"],
        "overall_status": resources["overall_status"],
        "query_timestamp": resources["query_timestamp"]
    }


def allocate_icu_bed(
    patient_risk_score: float,
    required_unit: str = "any",
    required_equipment: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Allocates ICU bed based on risk score.

    Args:
        patient_risk_score: Mortality risk (0-1)
        required_unit: Preferred unit (MICU, SICU, any)
        required_equipment: Equipment needed

    Returns:
        Allocation details
    """
    try:
        # Check current availability
        availability = check_bed_availability()

        if availability["icu_beds"]["available"] == 0:
            return {
                "bed_allocated": False,
                "reason": "No ICU beds available",
                "recommendation": "Transfer or queue patient"
            }

        # Simulate allocation (production would update database)
        bed_number = 7 + availability["icu_beds"]["occupied"]
        unit = required_unit if required_unit != "any" else "MICU"

        return {
            "bed_allocated": True,
            "allocation_details": {
                "unit": unit,
                "bed_number": f"Bed {bed_number}",
                "risk_score": patient_risk_score,
                "priority": "HIGH" if patient_risk_score > 0.7 else "MODERATE"
            },
            "equipment_assigned": required_equipment or [],
            "allocation_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Allocation failed: {str(e)}")
        return {"bed_allocated": False, "error": str(e)}
