"""
Anonymization Validation Tools
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# HIPAA Safe Harbor - 18 identifiers that must be removed
HIPAA_IDENTIFIERS = [
    'subject_id', 'hadm_id', 'stay_id', 'name', 'ssn', 'mrn',
    'phone', 'email', 'address', 'dob', 'admission_date',
    'discharge_date', 'death_date', 'exact_age'
]


def validate_anonymization(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates anonymization before external transmission.

    Args:
        data: Dictionary to validate

    Returns:
        Validation result
    """
    try:
        violations = []
        warnings = []

        # Check for HIPAA identifiers
        for key in data.keys():
            if key.lower() in [id.lower() for id in HIPAA_IDENTIFIERS]:
                violations.append(f"HIPAA identifier found: {key}")

        # Check for exact ages >89
        if 'age_at_admission' in data:
            age = data['age_at_admission']
            if isinstance(age, int) and age > 89:
                violations.append("Exact age >89 is HIPAA identifier")

        # Check for re-identifying combinations
        if all(k in data for k in ['ethnicity', 'insurance', 'marital_status']):
            warnings.append("Combination may be re-identifying")

        is_compliant = len(violations) == 0

        if not is_compliant:
            logger.error(f"Anonymization FAILED: {violations}")
        else:
            logger.info(f"Anonymization validated")

        return {
            "is_anonymized": is_compliant,
            "hipaa_compliant": is_compliant,
            "violations": violations,
            "warnings": warnings,
            "validation_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return {"is_anonymized": False, "error": str(e)}
