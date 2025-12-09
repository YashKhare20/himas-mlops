"""
Data Mappings for HIMAS Hospital A

Maps common/user-friendly terms to actual database values.
Ensures all queries use valid values that exist in the federated tables.
"""

from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# VALID VALUES FROM DATABASE
# ============================================================================

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

VALID_ICU_TYPES = [
    "Cardiac ICU",
    "Medical ICU",
    "Mixed ICU (2 Units)",
    "Surgical ICU",
    "Neuro ICU",
    "Mixed ICU (3+ Units)",
    "Other ICU"
]

VALID_GENDERS = ["M", "F"]

VALID_EARLY_ICU_SCORES = [0, 1, 2, 3]

VALID_FIRST_CAREUNITS = [
    "Coronary Care Unit (CCU)",
    "Medical Intensive Care Unit (MICU)",
    "Trauma SICU (TSICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    "Intensive Care Unit (ICU)"
]

VALID_ADMISSION_LOCATIONS = [
    "CLINIC REFERRAL",
    "EMERGENCY ROOM",
    "INFORMATION NOT AVAILABLE",
    "PACU",
    "PHYSICIAN REFERRAL",
    "PROCEDURE SITE",
    "TRANSFER FROM HOSPITAL",
    "TRANSFER FROM SKILLED NURSING FACILITY",
    "WALK-IN/SELF REFERRAL",
    "INTERNAL TRANSFER TO OR FROM PSYCH",
    "AMBULATORY SURGERY TRANSFER"
]

VALID_INSURANCES = [
    "Medicare",
    "Private",
    "Other",
    "Medicaid",
    "No charge"
]

VALID_MARITAL_STATUSES = [
    "SINGLE",
    "MARRIED",
    "DIVORCED",
    "WIDOWED"
]

# ============================================================================
# MAPPING DICTIONARIES (User-friendly → DB values)
# ============================================================================

# Maps common terms to actual admission_type values
ADMISSION_TYPE_MAPPING = {
    # Emergency mappings
    "emergency": ["EW EMER.", "DIRECT EMER."],
    "emer": ["EW EMER.", "DIRECT EMER."],
    "ew emer": ["EW EMER."],
    "ew emer.": ["EW EMER."],
    "direct emer": ["DIRECT EMER."],
    "direct emer.": ["DIRECT EMER."],
    "er": ["EW EMER.", "DIRECT EMER."],

    # Urgent mappings
    "urgent": ["URGENT"],

    # Elective mappings
    "elective": ["ELECTIVE"],
    "scheduled": ["ELECTIVE"],
    "planned": ["ELECTIVE"],

    # Observation mappings
    "observation": ["OBSERVATION ADMIT", "EU OBSERVATION", "DIRECT OBSERVATION", "AMBULATORY OBSERVATION"],
    "observation admit": ["OBSERVATION ADMIT"],
    "eu observation": ["EU OBSERVATION"],
    "direct observation": ["DIRECT OBSERVATION"],
    "ambulatory observation": ["AMBULATORY OBSERVATION"],
    "obs": ["OBSERVATION ADMIT", "EU OBSERVATION"],

    # Same day surgery
    "same day": ["SURGICAL SAME DAY ADMISSION"],
    "same_day": ["SURGICAL SAME DAY ADMISSION"],
    "surgical same day": ["SURGICAL SAME DAY ADMISSION"],
    "surgical same day admission": ["SURGICAL SAME DAY ADMISSION"],
    "day surgery": ["SURGICAL SAME DAY ADMISSION"],
}

# Maps common terms to actual icu_type values
ICU_TYPE_MAPPING = {
    # Cardiac
    "cardiac": ["Cardiac ICU"],
    "cardiac icu": ["Cardiac ICU"],
    "ccu": ["Cardiac ICU"],
    "heart": ["Cardiac ICU"],

    # Medical
    "medical": ["Medical ICU"],
    "medical icu": ["Medical ICU"],
    "micu": ["Medical ICU"],

    # Surgical
    "surgical": ["Surgical ICU"],
    "surgical icu": ["Surgical ICU"],
    "sicu": ["Surgical ICU"],

    # Neuro
    "neuro": ["Neuro ICU"],
    "neuro icu": ["Neuro ICU"],
    "neurological": ["Neuro ICU"],
    "neurosurgical": ["Neuro ICU"],

    # Mixed
    "mixed": ["Mixed ICU (2 Units)", "Mixed ICU (3+ Units)"],
    "mixed icu": ["Mixed ICU (2 Units)", "Mixed ICU (3+ Units)"],

    # Other
    "other": ["Other ICU"],
    "general": ["Other ICU"],
}

# Maps common terms to admission locations
ADMISSION_LOCATION_MAPPING = {
    "emergency": ["EMERGENCY ROOM"],
    "emergency room": ["EMERGENCY ROOM"],
    "er": ["EMERGENCY ROOM"],
    "ed": ["EMERGENCY ROOM"],

    "transfer": ["TRANSFER FROM HOSPITAL", "TRANSFER FROM SKILLED NURSING FACILITY"],
    "hospital transfer": ["TRANSFER FROM HOSPITAL"],
    "snf transfer": ["TRANSFER FROM SKILLED NURSING FACILITY"],
    "nursing home": ["TRANSFER FROM SKILLED NURSING FACILITY"],

    "clinic": ["CLINIC REFERRAL"],
    "clinic referral": ["CLINIC REFERRAL"],

    "physician": ["PHYSICIAN REFERRAL"],
    "physician referral": ["PHYSICIAN REFERRAL"],
    "doctor referral": ["PHYSICIAN REFERRAL"],

    "walk-in": ["WALK-IN/SELF REFERRAL"],
    "walk in": ["WALK-IN/SELF REFERRAL"],
    "self referral": ["WALK-IN/SELF REFERRAL"],

    "pacu": ["PACU"],
    "procedure": ["PROCEDURE SITE"],
    "surgery": ["AMBULATORY SURGERY TRANSFER"],
}


# ============================================================================
# MAPPING FUNCTIONS
# ============================================================================

def map_admission_type(user_input: str) -> Tuple[List[str], bool]:
    """
    Maps user-friendly admission type to valid database values.

    Args:
        user_input: User's description of admission type

    Returns:
        Tuple of (list of valid DB values, whether mapping was found)

    Example:
        >>> map_admission_type("emergency")
        (['EW EMER.', 'DIRECT EMER.'], True)
        >>> map_admission_type("URGENT")
        (['URGENT'], True)
    """
    if not user_input:
        return ["EW EMER."], False  # Default to emergency

    normalized = user_input.lower().strip()

    # Direct match with valid values (case-insensitive)
    for valid in VALID_ADMISSION_TYPES:
        if normalized == valid.lower():
            logger.info(
                f"Admission type direct match: '{user_input}' → '{valid}'")
            return [valid], True

    # Check mapping dictionary
    if normalized in ADMISSION_TYPE_MAPPING:
        mapped = ADMISSION_TYPE_MAPPING[normalized]
        logger.info(f"Admission type mapped: '{user_input}' → {mapped}")
        return mapped, True

    # Partial match
    for key, values in ADMISSION_TYPE_MAPPING.items():
        if key in normalized or normalized in key:
            logger.info(
                f"Admission type partial match: '{user_input}' → {values}")
            return values, True

    # No match found - default to emergency with warning
    logger.warning(
        f"Unknown admission type '{user_input}', defaulting to emergency types")
    return ["EW EMER.", "DIRECT EMER."], False


def map_icu_type(user_input: str) -> Tuple[List[str], bool]:
    """
    Maps user-friendly ICU type to valid database values.

    Args:
        user_input: User's description of ICU type

    Returns:
        Tuple of (list of valid DB values, whether mapping was found)
    """
    if not user_input:
        return VALID_ICU_TYPES, False  # All types if not specified

    normalized = user_input.lower().strip()

    # Direct match
    for valid in VALID_ICU_TYPES:
        if normalized == valid.lower():
            logger.info(f"ICU type direct match: '{user_input}' → '{valid}'")
            return [valid], True

    # Check mapping
    if normalized in ICU_TYPE_MAPPING:
        mapped = ICU_TYPE_MAPPING[normalized]
        logger.info(f"ICU type mapped: '{user_input}' → {mapped}")
        return mapped, True

    # Partial match
    for key, values in ICU_TYPE_MAPPING.items():
        if key in normalized or normalized in key:
            logger.info(f"ICU type partial match: '{user_input}' → {values}")
            return values, True

    logger.warning(f"Unknown ICU type '{user_input}', using all types")
    return VALID_ICU_TYPES, False


def map_admission_location(user_input: str) -> Tuple[List[str], bool]:
    """
    Maps user-friendly admission location to valid database values.
    """
    if not user_input:
        return VALID_ADMISSION_LOCATIONS, False

    normalized = user_input.lower().strip()

    # Direct match
    for valid in VALID_ADMISSION_LOCATIONS:
        if normalized == valid.lower():
            return [valid], True

    # Check mapping
    if normalized in ADMISSION_LOCATION_MAPPING:
        return ADMISSION_LOCATION_MAPPING[normalized], True

    # Partial match
    for key, values in ADMISSION_LOCATION_MAPPING.items():
        if key in normalized or normalized in key:
            return values, True

    logger.warning(f"Unknown admission location '{user_input}'")
    return VALID_ADMISSION_LOCATIONS, False


def validate_gender(gender: str) -> str:
    """
    Validates and normalizes gender value.

    Args:
        gender: Input gender value

    Returns:
        Valid gender value (M or F)
    """
    if not gender:
        return None

    normalized = gender.upper().strip()

    if normalized in ["M", "MALE", "MAN"]:
        return "M"
    elif normalized in ["F", "FEMALE", "WOMAN"]:
        return "F"
    else:
        logger.warning(f"Unknown gender '{gender}', returning None")
        return None


def validate_early_icu_score(score: int) -> int:
    """
    Validates early ICU score is in valid range (0-3).

    Args:
        score: Input score

    Returns:
        Valid score clamped to 0-3 range
    """
    if score is None:
        return 2  # Default middle value

    clamped = max(0, min(3, int(score)))

    if clamped != score:
        logger.warning(f"Early ICU score {score} clamped to {clamped}")

    return clamped


def get_emergency_admission_types() -> List[str]:
    """Returns all admission types that indicate emergency admission."""
    return ["EW EMER.", "DIRECT EMER."]


def get_observation_admission_types() -> List[str]:
    """Returns all observation-related admission types."""
    return ["OBSERVATION ADMIT", "EU OBSERVATION", "DIRECT OBSERVATION", "AMBULATORY OBSERVATION"]


def is_emergency_admission(admission_type: str) -> bool:
    """Checks if admission type indicates emergency."""
    if not admission_type:
        return False
    return admission_type in get_emergency_admission_types()


def format_admission_type_for_query(admission_types: List[str]) -> str:
    """
    Formats admission types for SQL-like query conditions.

    Args:
        admission_types: List of valid admission types

    Returns:
        Formatted string for query (e.g., "'EW EMER.' OR 'DIRECT EMER.'")
    """
    if not admission_types:
        return "'EW EMER.'"

    if len(admission_types) == 1:
        return f"'{admission_types[0]}'"

    return " OR ".join([f"'{t}'" for t in admission_types])


# ============================================================================
# HELPER FUNCTIONS FOR ANONYMIZATION
# ============================================================================

def get_admission_type_category(admission_type: str) -> str:
    """
    Returns a generalized category for admission type (for k-anonymity).

    Args:
        admission_type: Specific admission type

    Returns:
        Generalized category (EMERGENCY, URGENT, ELECTIVE, OBSERVATION, OTHER)
    """
    if not admission_type:
        return "UNKNOWN"

    normalized = admission_type.upper()

    if "EMER" in normalized:
        return "EMERGENCY"
    elif normalized == "URGENT":
        return "URGENT"
    elif normalized == "ELECTIVE":
        return "ELECTIVE"
    elif "OBSERVATION" in normalized or "OBS" in normalized:
        return "OBSERVATION"
    elif "SAME DAY" in normalized or "SURGICAL" in normalized:
        return "SURGICAL"
    else:
        return "OTHER"


def get_icu_type_category(icu_type: str) -> str:
    """
    Returns a generalized category for ICU type (for k-anonymity).
    """
    if not icu_type:
        return "UNKNOWN"

    normalized = icu_type.lower()

    if "cardiac" in normalized or "ccu" in normalized or "cvicu" in normalized:
        return "CARDIAC"
    elif "medical" in normalized or "micu" in normalized:
        return "MEDICAL"
    elif "surgical" in normalized or "sicu" in normalized or "trauma" in normalized:
        return "SURGICAL"
    elif "neuro" in normalized:
        return "NEURO"
    elif "mixed" in normalized:
        return "MIXED"
    else:
        return "OTHER"


# ============================================================================
# VALIDATION SUMMARY
# ============================================================================

def validate_query_parameters(
    admission_type: str = None,
    icu_type: str = None,
    early_icu_score: int = None,
    gender: str = None
) -> dict:
    """
    Validates and maps all query parameters at once.

    Args:
        admission_type: User's admission type input
        icu_type: User's ICU type input
        early_icu_score: ICU severity score
        gender: Gender value

    Returns:
        Dictionary with validated/mapped parameters and any warnings
    """
    warnings = []

    # Map admission type
    mapped_admission_types, admission_found = map_admission_type(
        admission_type)
    if not admission_found and admission_type:
        warnings.append(
            f"Admission type '{admission_type}' not recognized, using: {mapped_admission_types}")

    # Map ICU type
    mapped_icu_types, icu_found = map_icu_type(icu_type)
    if not icu_found and icu_type:
        warnings.append(
            f"ICU type '{icu_type}' not recognized, using: {mapped_icu_types}")

    # Validate early ICU score
    validated_score = validate_early_icu_score(early_icu_score)

    # Validate gender
    validated_gender = validate_gender(gender)

    result = {
        "admission_types": mapped_admission_types,
        "admission_type_primary": mapped_admission_types[0] if mapped_admission_types else None,
        "admission_type_category": get_admission_type_category(mapped_admission_types[0]) if mapped_admission_types else "UNKNOWN",
        "icu_types": mapped_icu_types,
        "icu_type_primary": mapped_icu_types[0] if mapped_icu_types else None,
        "icu_type_category": get_icu_type_category(mapped_icu_types[0]) if mapped_icu_types else "UNKNOWN",
        "early_icu_score": validated_score,
        "gender": validated_gender,
        "warnings": warnings,
        "all_valid": len(warnings) == 0
    }

    if warnings:
        for w in warnings:
            logger.warning(w)

    logger.info(
        f"Parameter validation complete: admission={result['admission_type_category']}, icu={result['icu_type_category']}, score={validated_score}")

    return result
