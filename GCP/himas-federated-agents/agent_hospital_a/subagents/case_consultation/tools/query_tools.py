"""
Cross-Hospital Query Tools via A2A Protocol

Communicates with the Federated Coordinator using A2A JSON-RPC protocol.
All queries use anonymized patient data only - never raw PII.
Uses validated/mapped values that match actual database entries.
"""
import requests
import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import os

from .privacy_tools import (
    anonymize_patient_data,
    verify_no_pii_in_request,
    log_privacy_audit
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Configuration
FEDERATED_COORDINATOR_URL = os.getenv(
    'FEDERATED_COORDINATOR_URL', 'http://localhost:8001')
HOSPITAL_ID = os.getenv('HOSPITAL_ID', 'hospital_a')


def _send_a2a_request(message_text: str, context_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Sends a request to the Federated Coordinator via A2A JSON-RPC protocol.

    Args:
        message_text: The query message to send
        context_id: Optional context ID for conversation continuity

    Returns:
        Parsed response from coordinator
    """
    message_id = f"msg-{uuid.uuid4().hex[:12]}"
    request_id = str(uuid.uuid4())

    logger.info("=" * 60)
    logger.info("SENDING A2A REQUEST TO FEDERATED COORDINATOR")
    logger.info("=" * 60)
    logger.info(f"Coordinator URL: {FEDERATED_COORDINATOR_URL}")
    logger.info(f"Message ID: {message_id}")

    # Build A2A JSON-RPC request
    a2a_request = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "id": request_id,
        "params": {
            "message": {
                "messageId": message_id,
                "role": "user",
                "parts": [{"text": message_text}]
            }
        }
    }

    # Add context if continuing conversation
    if context_id:
        a2a_request["params"]["contextId"] = context_id

    logger.info(f"Request payload prepared (JSON-RPC 2.0)")

    try:
        response = requests.post(
            FEDERATED_COORDINATOR_URL,
            json=a2a_request,
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        logger.info(f"Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # Extract the agent's response text
            if "result" in result:
                task_result = result["result"]

                # Get context ID for future requests
                new_context_id = task_result.get("contextId")

                # Extract response text from artifacts or history
                response_text = ""

                if "artifacts" in task_result:
                    for artifact in task_result["artifacts"]:
                        for part in artifact.get("parts", []):
                            if part.get("kind") == "text":
                                response_text += part.get("text", "")

                # Fallback to history if no artifacts
                if not response_text and "history" in task_result:
                    for msg in task_result["history"]:
                        if msg.get("role") == "agent":
                            for part in msg.get("parts", []):
                                if part.get("kind") == "text":
                                    response_text = part.get("text", "")

                logger.info("✓ A2A request successful")
                logger.info(f"Response preview: {response_text[:100]}..." if len(
                    response_text) > 100 else f"Response: {response_text}")

                return {
                    "success": True,
                    "response_text": response_text,
                    "context_id": new_context_id,
                    "task_id": task_result.get("id"),
                    "status": task_result.get("status", {}).get("state", "unknown")
                }

            elif "error" in result:
                error_msg = result["error"].get("message", "Unknown error")
                logger.error(f"A2A error response: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg
                }

        else:
            logger.error(f"HTTP error: {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }

    except requests.exceptions.Timeout:
        logger.error("Request timeout - coordinator may be overloaded")
        return {"success": False, "error": "Request timeout"}
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        return {"success": False, "error": f"Cannot connect to coordinator: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"success": False, "error": str(e)}


def query_peer_hospitals(
    required_capability: str,
    patient_data: Dict[str, Any],
    urgency: str = "routine"
) -> Dict[str, Any]:
    """
    Queries peer hospitals for capabilities via A2A protocol (privacy-preserved).

    This function:
    1. Anonymizes patient data (removes all PII)
    2. Verifies no PII leakage
    3. Sends anonymized query via A2A to Federated Coordinator
    4. Logs audit trail for HIPAA compliance

    Args:
        required_capability: Capability needed (e.g., "advanced_cardiac_care")
        patient_data: Raw patient data (will be anonymized)
        urgency: Query urgency (routine, urgent, emergency)

    Returns:
        Aggregated responses from peer hospitals
    """
    query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("=" * 70)
    logger.info(f"PEER HOSPITAL QUERY: {query_id}")
    logger.info(f"Required Capability: {required_capability}")
    logger.info(f"Urgency: {urgency}")
    logger.info("=" * 70)

    try:
        # Step 1: Anonymize patient data (this logs PII removal)
        logger.info("\n[STEP 1] ANONYMIZING PATIENT DATA...")
        patient_anonymized = anonymize_patient_data(patient_data)

        # Step 2: Build query message using VALIDATED VALUES
        # These are actual DB values, not user-friendly terms
        age_bucket = patient_anonymized.get("age_bucket", "unknown")
        admission_type = patient_anonymized.get(
            "admission_type", "EW EMER.")  # Actual DB value
        admission_category = patient_anonymized.get(
            "admission_type_category", "EMERGENCY")  # For description
        early_icu_score = patient_anonymized.get("early_icu_score", 2)
        icu_type = patient_anonymized.get("icu_type")
        icu_category = patient_anonymized.get("icu_type_category", "UNKNOWN")

        # Log the mapped values
        logger.info(
            f"Using validated admission_type: '{admission_type}' (category: {admission_category})")
        if icu_type:
            logger.info(
                f"Using validated icu_type: '{icu_type}' (category: {icu_category})")

        query_message = f"""
Query from {HOSPITAL_ID} ({urgency} priority):

I need to find hospitals with '{required_capability}' capability for a patient consultation.

Anonymized Patient Profile:
- Age bucket: {age_bucket}
- Admission type: {admission_type}
- Admission category: {admission_category}
- Early ICU score: {early_icu_score}
- ICU type: {icu_type if icu_type else 'Not specified'}
- Risk score range: {patient_anonymized.get('risk_score_range', '0.5-0.6')}

Please use query_hospital_capabilities to find hospitals with '{required_capability}' 
and query_similar_cases with:
- age_bucket: "{age_bucket}"
- admission_type: "{admission_type}"
- early_icu_score: {early_icu_score}

Requesting hospital: {HOSPITAL_ID}
"""

        # Step 3: Verify no PII in outgoing request
        logger.info("\n[STEP 2] VERIFYING NO PII IN REQUEST...")
        verify_no_pii_in_request({
            "message": query_message,
            "patient_data": patient_anonymized
        })

        # Step 4: Send A2A request
        logger.info("\n[STEP 3] SENDING A2A REQUEST TO COORDINATOR...")
        a2a_response = _send_a2a_request(query_message)

        # Step 5: Create audit log
        logger.info("\n[STEP 4] CREATING AUDIT LOG...")

        # Enrich patient_anonymized with original risk_score for audit logging
        patient_anonymized_for_audit = {
            **patient_anonymized,
            # Original numeric value
            "risk_score": patient_data.get("risk_score"),
            "user_role": "clinical_staff"
        }
        audit_entry = log_privacy_audit(
            operation="peer_hospital_query",
            source_hospital=HOSPITAL_ID,
            target="federated_coordinator",
            data_fields_shared=list(patient_anonymized.keys()),
            pii_removed=patient_anonymized.get("removed_fields", []),
            patient_anonymized=patient_anonymized_for_audit
        )

        # Step 6: Process response
        if a2a_response.get("success"):
            response_text = a2a_response.get("response_text", "")

            logger.info("\n[STEP 5] QUERY COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)

            return {
                "query_id": query_id,
                "capability_queried": required_capability,
                "success": True,
                "coordinator_response": response_text,
                "context_id": a2a_response.get("context_id"),
                "privacy_verified": True,
                "audit_id": audit_entry.get("audit_id"),
                "patient_anonymized": patient_anonymized,
                "timestamp": datetime.now().isoformat()
            }
        else:
            error_msg = a2a_response.get("error", "Unknown error")
            logger.error(f"\n[ERROR] Query failed: {error_msg}")

            return {
                "query_id": query_id,
                "success": False,
                "error": error_msg,
                "fallback_recommendation": "Manual consultation with peer hospitals recommended",
                "audit_id": audit_entry.get("audit_id")
            }

    except Exception as e:
        logger.error(f"Peer hospital query failed: {str(e)}")
        return {
            "query_id": query_id,
            "success": False,
            "error": str(e),
            "capability_found": False,
            "fallback_recommendation": "Manual consultation recommended due to system error"
        }


def query_similar_cases(
    patient_data: Dict[str, Any],
    requesting_hospital: Optional[str] = None
) -> Dict[str, Any]:
    """
    Queries peer hospitals for similar case outcomes via A2A protocol.

    Args:
        patient_data: Raw patient data (will be anonymized)
        requesting_hospital: Hospital making request

    Returns:
        Privacy-preserved aggregate statistics
    """
    query_id = f"cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    hospital_id = requesting_hospital or HOSPITAL_ID

    logger.info("=" * 70)
    logger.info(f"SIMILAR CASES QUERY: {query_id}")
    logger.info("=" * 70)

    try:
        # Anonymize patient data
        logger.info("\n[STEP 1] ANONYMIZING PATIENT DATA...")
        patient_anonymized = anonymize_patient_data(patient_data)

        # Build query using VALIDATED values from anonymization
        age_bucket = patient_anonymized.get("age_bucket", "65-70")
        admission_type = patient_anonymized.get(
            "admission_type", "EW EMER.")  # Actual DB value
        admission_category = patient_anonymized.get(
            "admission_type_category", "EMERGENCY")
        early_icu_score = patient_anonymized.get("early_icu_score", 2)

        logger.info(
            f"Query parameters: age={age_bucket}, admission={admission_type}, score={early_icu_score}")

        query_message = f"""
Similar case query from {hospital_id}:

Please use query_similar_cases with these anonymized parameters:
- age_bucket: "{age_bucket}"
- admission_type: "{admission_type}"
- early_icu_score: {early_icu_score}
- requesting_hospital: "{hospital_id}"

Note: admission_type "{admission_type}" is a {admission_category} category admission.

Return privacy-preserved survival statistics from peer hospitals.
"""

        # Verify no PII
        logger.info("\n[STEP 2] VERIFYING NO PII...")
        verify_no_pii_in_request({"message": query_message})

        # Send A2A request
        logger.info("\n[STEP 3] SENDING A2A REQUEST...")
        a2a_response = _send_a2a_request(query_message)

        # Enrich for audit
        patient_anonymized_for_audit = {
            **patient_anonymized,
            "risk_score": patient_data.get("risk_score"),
            "user_role": "clinical_staff"
        }

        # Audit log
        audit_entry = log_privacy_audit(
            operation="similar_cases_query",
            source_hospital=hospital_id,
            target="federated_coordinator",
            data_fields_shared=["age_bucket",
                                "admission_type", "early_icu_score"],
            pii_removed=patient_anonymized.get("removed_fields", []),
            patient_anonymized=patient_anonymized_for_audit
        )

        if a2a_response.get("success"):
            logger.info("\n✓ Similar cases query successful")
            return {
                "query_id": query_id,
                "success": True,
                "coordinator_response": a2a_response.get("response_text"),
                "context_id": a2a_response.get("context_id"),
                "query_parameters": {
                    "age_bucket": age_bucket,
                    "admission_type": admission_type,
                    "early_icu_score": early_icu_score
                },
                "privacy_verified": True,
                "audit_id": audit_entry.get("audit_id")
            }
        else:
            return {
                "query_id": query_id,
                "success": False,
                "error": a2a_response.get("error"),
                "fallback": "Use local historical data for comparison"
            }

    except Exception as e:
        logger.error(f"Similar cases query failed: {str(e)}")
        return {"success": False, "error": str(e)}


def initiate_transfer(
    target_hospital: str,
    patient_data: Dict[str, Any],
    required_capability: str,
    urgency: str = "urgent"
) -> Dict[str, Any]:
    """
    Initiates patient transfer to peer hospital via A2A protocol.

    Now also:
    - Sends email notification to receiving hospital
    - Logs to receiving hospital's audit table
    """
    transfer_id = f"transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("=" * 70)
    logger.info(f"TRANSFER INITIATION: {transfer_id}")
    logger.info(f"Target: {target_hospital}")
    logger.info(f"Urgency: {urgency}")
    logger.info("=" * 70)

    try:
        # Step 1: Anonymize patient data
        logger.info("\n[STEP 1] ANONYMIZING PATIENT DATA FOR TRANSFER...")
        patient_anonymized = anonymize_patient_data(patient_data)

        # Create patient fingerprint
        from .privacy_tools import create_query_fingerprint
        patient_fingerprint = create_query_fingerprint(patient_anonymized)

        # Build transfer request message
        transfer_message = f"""
Transfer request from {HOSPITAL_ID}:

Please use initiate_transfer with these parameters:
- source_hospital: "{HOSPITAL_ID}"
- target_hospital: "{target_hospital}"
- patient_fingerprint: "{patient_fingerprint}"
- transfer_reason: "{required_capability}"
- urgency: "{urgency.upper()}"

Patient profile (anonymized):
- Age bucket: {patient_anonymized.get('age_bucket')}
- Admission type: {patient_anonymized.get('admission_type')}
- Risk score range: {patient_anonymized.get('risk_score_range')}

Please confirm bed availability and initiate transfer coordination.
"""

        # Step 2: Verify no PII
        logger.info("\n[STEP 2] VERIFYING NO PII IN TRANSFER REQUEST...")
        verify_no_pii_in_request({
            "message": transfer_message,
            "patient_fingerprint": patient_fingerprint
        })

        # Step 3: Send A2A request
        logger.info("\n[STEP 3] SENDING TRANSFER REQUEST VIA A2A...")
        a2a_response = _send_a2a_request(transfer_message)

        # Step 4: Audit log at SOURCE hospital
        logger.info("\n[STEP 4] CREATING AUDIT LOG (SOURCE HOSPITAL)...")

        # Enrich patient_anonymized with original risk_score for audit
        patient_anonymized_for_audit = {
            **patient_anonymized,
            "risk_score": patient_data.get("risk_score"),
            "user_role": "clinical_staff"
        }

        audit_entry = log_privacy_audit(
            operation="transfer_initiation",
            source_hospital=HOSPITAL_ID,
            target=target_hospital,
            data_fields_shared=["age_bucket", "admission_type",
                                "risk_score_range", "fingerprint"],
            pii_removed=patient_anonymized.get("removed_fields", []),
            patient_anonymized=patient_anonymized_for_audit
        )

        if a2a_response.get("success"):
            # Extract bed reservation from response (parse if needed)
            # For now, use placeholder - in production, parse from coordinator response
            bed_reservation = {
                "bed_id": "ICU_BED_1",
                "unit": "MICU",
                "reserved_until": datetime.now().replace(
                    hour=(datetime.now().hour + 4) % 24
                ).isoformat()
            }
            estimated_transport_minutes = 45  # Would come from coordinator

            # Step 5: Log to RECEIVING hospital's audit table
            logger.info(
                "\n[STEP 5] LOGGING TO RECEIVING HOSPITAL AUDIT TABLE...")
            try:
                from ...privacy_guardian.tools.audit_logging import log_transfer_receipt

                receipt_result = log_transfer_receipt(
                    transfer_id=transfer_id,
                    source_hospital=HOSPITAL_ID,
                    receiving_hospital=target_hospital,
                    patient_anonymized=patient_anonymized_for_audit,
                    transfer_reason=required_capability,
                    urgency=urgency.upper(),
                    bed_reservation=bed_reservation
                )
                logger.info(
                    f"✓ Receiving hospital audit: {receipt_result.get('log_id')}")

            except ImportError as e:
                logger.warning(f"Could not import log_transfer_receipt: {e}")
                receipt_result = {"error": str(e)}
            except Exception as e:
                logger.error(f"Failed to log to receiving hospital: {e}")
                receipt_result = {"error": str(e)}

            # Step 6: Send email notification
            logger.info("\n[STEP 6] SENDING EMAIL NOTIFICATION...")
            try:
                from ...privacy_guardian.tools.notification_tools import send_transfer_notification

                # Extract risk level
                risk_score = patient_data.get("risk_score", 0.5)
                if risk_score >= 0.7:
                    risk_level = "HIGH"
                elif risk_score >= 0.3:
                    risk_level = "MODERATE"
                else:
                    risk_level = "LOW"

                notification_result = send_transfer_notification(
                    transfer_id=transfer_id,
                    source_hospital=HOSPITAL_ID,
                    target_hospital=target_hospital,
                    transfer_reason=required_capability,
                    urgency=urgency.upper(),
                    patient_age_bucket=patient_anonymized.get(
                        "age_bucket", "unknown"),
                    risk_level=risk_level,
                    bed_reservation=bed_reservation,
                    estimated_transport_minutes=estimated_transport_minutes
                )

                if notification_result.get("sent"):
                    logger.info(
                        f"✓ Email notification sent to {notification_result.get('recipients_count')} recipients")
                else:
                    logger.warning(
                        f"Email notification not sent: {notification_result.get('reason')}")

            except ImportError as e:
                logger.warning(f"Could not import notification_tools: {e}")
                notification_result = {"sent": False, "reason": str(e)}
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
                notification_result = {"sent": False, "reason": str(e)}

            logger.info("\n✓ TRANSFER REQUEST SUBMITTED SUCCESSFULLY")
            logger.info("=" * 70)

            return {
                "transfer_id": transfer_id,
                "transfer_confirmed": True,
                "source_hospital": HOSPITAL_ID,
                "target_hospital": target_hospital,
                "coordinator_response": a2a_response.get("response_text"),
                "context_id": a2a_response.get("context_id"),
                "patient_fingerprint": patient_fingerprint[:16] + "...",
                "urgency": urgency,
                "privacy_verified": True,
                "audit_id": audit_entry.get("audit_id"),
                "receiving_hospital_audit": receipt_result,
                "email_notification": notification_result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            logger.error(
                f"\n✗ Transfer request failed: {a2a_response.get('error')}")
            return {
                "transfer_id": transfer_id,
                "transfer_confirmed": False,
                "error": a2a_response.get("error"),
                "fallback": "Contact target hospital directly via phone"
            }

    except Exception as e:
        logger.error(f"Transfer initiation failed: {str(e)}")
        return {
            "transfer_id": transfer_id,
            "transfer_confirmed": False,
            "error": str(e)
        }


def get_transfer_status(transfer_id: str) -> Dict[str, Any]:
    """
    Gets the status of an active transfer via A2A.

    Args:
        transfer_id: The transfer ID to look up

    Returns:
        Current transfer status
    """
    logger.info(f"Checking transfer status: {transfer_id}")

    query_message = f"""
Please use get_transfer_status to check the status of transfer: "{transfer_id}"
"""

    a2a_response = _send_a2a_request(query_message)

    if a2a_response.get("success"):
        return {
            "transfer_id": transfer_id,
            "success": True,
            "coordinator_response": a2a_response.get("response_text")
        }
    else:
        return {
            "transfer_id": transfer_id,
            "success": False,
            "error": a2a_response.get("error")
        }
