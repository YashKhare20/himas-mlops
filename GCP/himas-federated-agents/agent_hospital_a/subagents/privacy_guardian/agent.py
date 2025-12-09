"""
Privacy Guardian Agent
HIPAA compliance, audit logging, and data anonymization oversight.

Uses data_mappings module for validating anonymized categories.
"""

from google.genai import types
from google.adk import Agent

from ...config import config
from .prompt import PRIVACY_GUARDIAN_INSTRUCTION
from .tools.audit_logging import (
    log_prediction_access,
    log_external_query,
    get_audit_trail,
    log_transfer_receipt
)
from .tools.anonymization import validate_anonymization

from .tools.notification_tools import (
    send_transfer_notification,
    send_test_notification
)


privacy_guardian_agent = Agent(
    model=config.MODEL_NAME,
    name="privacy_guardian_agent",
    description="""
    "Privacy compliance specialist ensuring HIPAA compliance, data anonymization, "
    "audit logging at both source and receiving hospitals, PII protection, and "
    "email notifications for patient transfers. "
    "Implements k-anonymity (k≥5) and differential privacy (ε=0.1)."

    Privacy Features:
    - Validates all patient data is properly anonymized before external sharing
    - Uses generalized categories (EMERGENCY, CARDIAC, etc.) instead of exact values
    - Logs all data access with full audit trail
    - Enforces k-anonymity (k≥5) and differential privacy (ε=0.1)

    Category Mappings Used:
    - Admission types → EMERGENCY, URGENT, ELECTIVE, OBSERVATION, SURGICAL
    - ICU types → CARDIAC, MEDICAL, SURGICAL, NEURO, MIXED
    """,
    instruction=PRIVACY_GUARDIAN_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(
            temperature=0.1,  # Low temperature for consistent privacy decisions
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                ),
            ]
        ),
    tools=[
        log_prediction_access,
        log_external_query,
        log_transfer_receipt,
        get_audit_trail,
        validate_anonymization,
        send_transfer_notification,
        send_test_notification
    ],
)