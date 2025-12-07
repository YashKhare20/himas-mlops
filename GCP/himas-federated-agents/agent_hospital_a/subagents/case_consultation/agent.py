"""
Case Consultation Sub-Agent for Hospital A
Handles cross-hospital consultations and transfers via A2A protocol.
All communications are privacy-preserved with HIPAA Safe Harbor de-identification.

Uses data_mappings module for converting user-friendly terms to database values.
"""
import logging
from typing import Dict, Any, Optional
from google.adk.agents import Agent
from google.genai import types

from ...config import config
from ...data_mappings import (
    map_admission_type,
    map_icu_type,
    validate_early_icu_score,
    get_admission_type_category,
    get_icu_type_category,
    VALID_ADMISSION_TYPES,
    VALID_ICU_TYPES
)
from .prompt import CASE_CONSULTATION_INSTRUCTION

# Import the A2A-enabled tools
from .tools.query_tools import (
    query_peer_hospitals as _query_peer_hospitals,
    query_similar_cases as _query_similar_cases,
    initiate_transfer as _initiate_transfer,
    get_transfer_status as _get_transfer_status
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def query_peer_hospitals_for_capability(
    required_capability: str,
    age_at_admission: int,
    admission_type: str,
    early_icu_score: int,
    risk_score: float,
    urgency: str = "routine",
    icu_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query peer hospitals for a specific capability (privacy-preserved via A2A).

    This function anonymizes all patient data before querying the federated network.
    No PII is ever transmitted to peer hospitals.

    Admission type and ICU type are automatically mapped from user-friendly terms
    to database values using the data_mappings module.

    Args:
        required_capability: Capability needed (e.g., "advanced_cardiac_care", "ecmo", 
                           "infectious_disease", "cardiac_surgery")
        age_at_admission: Patient age (will be bucketed to 5-year ranges)
        admission_type: Type of admission - user-friendly terms accepted:
                       "emergency", "urgent", "elective", "observation", etc.
        early_icu_score: ICU severity score 0-3 (3=most severe)
        risk_score: Mortality risk score 0-1 (will be bucketed)
        urgency: Query urgency (routine, urgent, emergency)
        icu_type: Optional ICU type filter - user-friendly terms accepted:
                 "cardiac", "medical", "surgical", "neuro", "mixed"

    Returns:
        dict containing:
        - hospitals_with_capability: List of hospitals with the capability
        - survival_statistics: Privacy-preserved outcome statistics
        - recommended_transfer_target: Best hospital for transfer
        - privacy_verified: Confirmation that no PII was transmitted
        - data_mappings_applied: Shows how user terms were mapped
    """
    logger.info(
        f"[CaseConsultation] Querying peer hospitals for: {required_capability}")

    # Map user-friendly terms to database values
    db_admission_types, admission_mapped = map_admission_type(admission_type)
    admission_category = get_admission_type_category(db_admission_types[0])

    db_icu_types = None
    icu_category = None
    if icu_type:
        db_icu_types, icu_mapped = map_icu_type(icu_type)
        icu_category = get_icu_type_category(db_icu_types[0])

    # Validate early ICU score
    validated_score = validate_early_icu_score(early_icu_score)

    # Build patient data dict for anonymization
    patient_data = {
        "age_at_admission": int(age_at_admission),
        "admission_type": db_admission_types[0],  # Use primary mapped value
        "admission_type_category": admission_category,
        "early_icu_score": validated_score,
        "risk_score": float(risk_score),
        "icu_type": db_icu_types[0] if db_icu_types else None,
        "icu_type_category": icu_category
    }

    # Call the A2A query tool (handles anonymization internally)
    result = _query_peer_hospitals(
        required_capability=required_capability,
        patient_data=patient_data,
        urgency=urgency
    )

    # Add mapping information to result
    result["data_mappings_applied"] = {
        "admission_type_input": str(admission_type),
        "admission_type_mapped": db_admission_types,
        "admission_type_category": admission_category,
        "icu_type_input": str(icu_type) if icu_type else None,
        "icu_type_mapped": db_icu_types,
        "icu_type_category": icu_category,
        "early_icu_score_validated": validated_score
    }

    return result


def query_similar_cases_network(
    age_at_admission: int,
    admission_type: str,
    early_icu_score: int,
    risk_score: float,
    icu_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query the federated network for similar case outcomes (privacy-preserved).

    Returns aggregate statistics from peer hospitals without exposing individual
    patient data. Uses k-anonymity (k≥5) and differential privacy (ε=0.1).

    Admission type and ICU type are automatically mapped from user-friendly terms.

    Args:
        age_at_admission: Patient age (will be bucketed)
        admission_type: Type of admission - user-friendly terms accepted:
                       "emergency", "urgent", "elective", "observation"
        early_icu_score: ICU severity score 0-3
        risk_score: Mortality risk score 0.0-1.0 (will be bucketed for privacy)
        icu_type: Optional ICU type filter - user-friendly terms accepted:
                 "cardiac", "medical", "surgical", "neuro", "mixed"

    Returns:
        dict containing:
        - peer_hospital_results: Statistics from each peer hospital
        - aggregate_survival_rate: Network-wide survival rate
        - sample_sizes: Number of similar cases at each hospital
        - privacy_guarantees: Privacy methods applied
        - data_mappings_applied: Shows how user terms were mapped
    """
    logger.info("[CaseConsultation] Querying network for similar cases")

    # Map user-friendly terms to database values
    db_admission_types, _ = map_admission_type(admission_type)
    admission_category = get_admission_type_category(db_admission_types[0])

    db_icu_types = None
    icu_category = None
    if icu_type:
        db_icu_types, _ = map_icu_type(icu_type)
        icu_category = get_icu_type_category(db_icu_types[0])

    validated_score = validate_early_icu_score(early_icu_score)

    patient_data = {
        "age_at_admission": int(age_at_admission),
        "admission_type": db_admission_types[0],
        "admission_type_category": admission_category,
        "early_icu_score": validated_score,
        "risk_score": float(risk_score),
        "icu_type": db_icu_types[0] if db_icu_types else None,
        "icu_type_category": icu_category
    }

    result = _query_similar_cases(patient_data)

    # Add mapping information
    result["data_mappings_applied"] = {
        "admission_type_input": str(admission_type),
        "admission_type_mapped": db_admission_types,
        "admission_type_category": admission_category,
        "icu_type_input": str(icu_type) if icu_type else None,
        "icu_type_mapped": db_icu_types,
        "icu_type_category": icu_category,
        "early_icu_score_validated": validated_score,
        "risk_score_input": risk_score
    }

    return result


def initiate_patient_transfer(
    target_hospital: str,
    transfer_reason: str,
    age_at_admission: int,
    admission_type: str,
    early_icu_score: int,
    risk_score: float,
    urgency: str = "HIGH",
    icu_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Initiate a patient transfer to a peer hospital (privacy-preserved).

    Coordinates transfer via the Federated Coordinator using A2A protocol.
    Only anonymized patient data is shared with the receiving hospital.

    Admission type and ICU type are automatically mapped from user-friendly terms.

    Args:
        target_hospital: Target hospital ID (e.g., "hospital_b")
        transfer_reason: Clinical reason/capability needed
        age_at_admission: Patient age (will be bucketed)
        admission_type: Type of admission - user-friendly terms accepted
        early_icu_score: ICU severity score
        risk_score: Mortality risk score
        urgency: Transfer urgency (HIGH, MEDIUM, LOW)
        icu_type: Optional ICU type - user-friendly terms accepted

    Returns:
        dict containing:
        - transfer_confirmed: Boolean success indicator
        - transfer_id: Unique transfer identifier
        - bed_reservation: Reserved bed details
        - estimated_transport_minutes: ETA
        - receiving_team_notified: Boolean
        - data_mappings_applied: Shows how user terms were mapped
    """
    logger.info(f"[CaseConsultation] Initiating transfer to {target_hospital}")

    # Map user-friendly terms
    db_admission_types, _ = map_admission_type(admission_type)
    admission_category = get_admission_type_category(db_admission_types[0])

    db_icu_types = None
    icu_category = None
    if icu_type:
        db_icu_types, _ = map_icu_type(icu_type)
        icu_category = get_icu_type_category(db_icu_types[0])

    validated_score = validate_early_icu_score(early_icu_score)

    patient_data = {
        "age_at_admission": int(age_at_admission),
        "admission_type": db_admission_types[0],
        "admission_type_category": admission_category,
        "early_icu_score": validated_score,
        "risk_score": float(risk_score),
        "icu_type": db_icu_types[0] if db_icu_types else None,
        "icu_type_category": icu_category
    }

    result = _initiate_transfer(
        target_hospital=target_hospital,
        patient_data=patient_data,
        required_capability=transfer_reason,
        urgency=urgency
    )

    # Add mapping information
    result["data_mappings_applied"] = {
        "admission_type_input": str(admission_type),
        "admission_type_mapped": db_admission_types,
        "icu_type_input": str(icu_type) if icu_type else None,
        "icu_type_mapped": db_icu_types
    }

    return result


def check_transfer_status(transfer_id: str) -> Dict[str, Any]:
    """
    Check the status of an active transfer.

    Args:
        transfer_id: The transfer ID to look up

    Returns:
        dict containing current transfer status and details
    """
    logger.info(f"[CaseConsultation] Checking transfer status: {transfer_id}")
    return _get_transfer_status(transfer_id)


def get_valid_admission_types() -> Dict[str, Any]:
    """
    Returns the list of valid admission types and their mappings.

    Use this to understand what admission type terms are accepted
    and how they map to database values.

    Returns:
        dict containing valid admission types and mapping examples
    """
    return {
        "valid_database_values": list(VALID_ADMISSION_TYPES),
        "user_friendly_mappings": {
            "emergency": ["EW EMER.", "DIRECT EMER."],
            "urgent": ["URGENT"],
            "elective": ["ELECTIVE"],
            "observation": ["OBSERVATION ADMIT", "EU OBSERVATION", "DIRECT OBSERVATION"],
            "same_day": ["SURGICAL SAME DAY ADMISSION"]
        },
        "note": "You can use either user-friendly terms or exact database values"
    }


def get_valid_icu_types() -> Dict[str, Any]:
    """
    Returns the list of valid ICU types and their mappings.

    Returns:
        dict containing valid ICU types and mapping examples
    """
    return {
        "valid_database_values": list(VALID_ICU_TYPES),
        "user_friendly_mappings": {
            "cardiac": ["Cardiac ICU"],
            "medical": ["Medical ICU"],
            "surgical": ["Surgical ICU"],
            "neuro": ["Neuro ICU"],
            "mixed": ["Mixed ICU (2 Units)", "Mixed ICU (3+ Units)"]
        },
        "note": "You can use either user-friendly terms or exact database values"
    }


case_consultation_agent = Agent(
    model=config.MODEL_NAME,
    name="case_consultation_agent",
    description="""
    "Cross-hospital consultation specialist for federated case queries, "
    "capability matching, and patient transfers. Coordinates with Privacy Guardian "
    "for data anonymization, audit logging, and email notifications. "
    "All operations are privacy-preserving and HIPAA compliant."
    
    Capabilities:
    - Query peer hospitals for specific capabilities (advanced_cardiac_care, ecmo, etc.)
    - Find similar case outcomes across the federated network
    - Initiate and track patient transfers
    
    Data Mapping Features:
    - Accepts user-friendly terms (e.g., "emergency", "cardiac")
    - Automatically maps to database values (e.g., "EW EMER.", "Cardiac ICU")
    - Returns mapping info in all responses for transparency
    
    Privacy Guarantees:
    - All patient data is anonymized before transmission (HIPAA Safe Harbor)
    - No PII is ever shared with peer hospitals
    - Results use k-anonymity (k≥5) and differential privacy (ε=0.1)
    - Complete audit trail for HIPAA compliance
    """,
    instruction=CASE_CONSULTATION_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    tools=[
        query_peer_hospitals_for_capability,
        query_similar_cases_network,
        initiate_patient_transfer,
        check_transfer_status,
        get_valid_admission_types,
        get_valid_icu_types
    ],
)
