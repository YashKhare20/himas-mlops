"""
Audit Logging Tools for HIMAS
Logs all data access to BigQuery audit tables for HIPAA compliance.

FIXED VERSION - Key changes:
1. Python logging connected to Cloud Logging at module load
2. BigQuery table auto-creation
3. Better error reporting
4. Singleton pattern for clients

Tables used:
- audit_logs.hospital_a_audit_log
- audit_logs.hospital_b_audit_log
- audit_logs.hospital_c_audit_log
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

GOOGLE_CLOUD_PROJECT = os.getenv(
    'GOOGLE_CLOUD_PROJECT', 'erudite-carving-472018-r5')
HOSPITAL_ID = os.getenv('HOSPITAL_ID', 'hospital_a')
AUDIT_LOG_DATASET = os.getenv('AUDIT_LOG_DATASET', 'audit_logs')
AUDIT_LOG_TABLE = f'{HOSPITAL_ID}_audit_log'
ENABLE_CLOUD_LOGGING = os.getenv(
    'ENABLE_CLOUD_LOGGING', 'true').lower() == 'true'

# ============================================================================
# CLOUD LOGGING SETUP - CONNECT PYTHON LOGGING TO CLOUD LOGGING
# ============================================================================

# Singleton clients to avoid recreating on every call
_cloud_logging_client = None
_bigquery_client = None
_table_verified = False


def _get_cloud_logging_client():
    """Get or create Cloud Logging client (singleton)."""
    global _cloud_logging_client
    if _cloud_logging_client is None:
        try:
            from google.cloud import logging as cloud_logging
            _cloud_logging_client = cloud_logging.Client(
                project=GOOGLE_CLOUD_PROJECT)
        except Exception as e:
            print(f"WARNING: Could not create Cloud Logging client: {e}")
    return _cloud_logging_client


def _get_bigquery_client():
    """Get or create BigQuery client (singleton)."""
    global _bigquery_client
    if _bigquery_client is None:
        try:
            from google.cloud import bigquery
            _bigquery_client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT)
        except Exception as e:
            print(f"WARNING: Could not create BigQuery client: {e}")
    return _bigquery_client


def setup_cloud_logging_integration():
    """
    Connect Python's logging module to Google Cloud Logging.

    This makes ALL logger.info(), logger.warning(), etc. calls
    appear in Cloud Logging automatically.

    Call this ONCE at application startup.
    """
    if not ENABLE_CLOUD_LOGGING:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        logging.info("Cloud Logging disabled. Using console only.")
        return False

    try:
        from google.cloud import logging as cloud_logging
        from google.cloud.logging_v2.handlers import CloudLoggingHandler, setup_logging

        client = _get_cloud_logging_client()
        if client is None:
            raise RuntimeError("Cloud Logging client not available")

        # Create handler that sends to Cloud Logging
        handler = CloudLoggingHandler(
            client,
            name=f"himas-{HOSPITAL_ID}-python",
            labels={
                "hospital_id": HOSPITAL_ID,
                "application": "himas-agent"
            }
        )
        handler.setLevel(logging.INFO)

        # Get root logger and configure it
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Clear existing handlers to avoid duplicates
        root_logger.handlers = []

        # Add Cloud Logging handler
        root_logger.addHandler(handler)

        # Also add console handler for local visibility
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        ))
        root_logger.addHandler(console_handler)

        # Use setup_logging for comprehensive integration
        setup_logging(handler, log_level=logging.INFO)

        logging.info("=" * 60)
        logging.info(f"✓ Cloud Logging connected for {HOSPITAL_ID}")
        logging.info(f"  Project: {GOOGLE_CLOUD_PROJECT}")
        logging.info(f"  Log name: himas-{HOSPITAL_ID}-python")
        logging.info("=" * 60)

        return True

    except ImportError:
        logging.warning(
            "google-cloud-logging not installed. Install with: pip install google-cloud-logging")
        logging.basicConfig(level=logging.INFO)
        return False
    except Exception as e:
        logging.warning(f"Failed to setup Cloud Logging: {e}")
        logging.basicConfig(level=logging.INFO)
        return False


# Initialize logging when module is imported
# This ensures ALL subsequent logger calls go to Cloud Logging
_cloud_logging_initialized = setup_cloud_logging_integration()

# Now create the module logger (it will use Cloud Logging if setup succeeded)
logger = logging.getLogger(__name__)


# ============================================================================
# BIGQUERY TABLE SETUP
# ============================================================================

def ensure_audit_table_exists() -> bool:
    """
    Ensure the BigQuery audit table exists, create if not.

    Returns:
        True if table exists/was created, False on error
    """
    global _table_verified

    if _table_verified:
        return True

    client = _get_bigquery_client()
    if client is None:
        logger.error("BigQuery client not available")
        return False

    try:
        from google.cloud import bigquery
        from google.cloud.exceptions import NotFound

        dataset_ref = f"{GOOGLE_CLOUD_PROJECT}.{AUDIT_LOG_DATASET}"
        table_ref = f"{dataset_ref}.{AUDIT_LOG_TABLE}"

        # Check/create dataset
        try:
            client.get_dataset(dataset_ref)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            dataset.description = "HIMAS Audit Logs for HIPAA Compliance"
            client.create_dataset(dataset)
            logger.info(f"✓ Created audit dataset: {AUDIT_LOG_DATASET}")

        # Check/create table
        try:
            client.get_table(table_ref)
            logger.info(f"✓ Audit table exists: {table_ref}")
            _table_verified = True
            return True
        except NotFound:
            pass

        # Create table with schema
        schema = [
            bigquery.SchemaField("log_id", "STRING", mode="REQUIRED",
                                 description="Unique audit log entry ID"),
            bigquery.SchemaField("hospital_id", "STRING", mode="REQUIRED",
                                 description="Hospital identifier"),
            bigquery.SchemaField("action", "STRING", mode="REQUIRED",
                                 description="Action performed"),
            bigquery.SchemaField("action_category", "STRING",
                                 description="Category: prediction, external_query, transfer"),
            bigquery.SchemaField("event_timestamp", "TIMESTAMP", mode="REQUIRED",
                                 description="When the event occurred"),
            bigquery.SchemaField("logged_at", "TIMESTAMP",
                                 description="When the log was created"),
            bigquery.SchemaField("user_id", "STRING", mode="REQUIRED",
                                 description="User who performed the action"),
            bigquery.SchemaField("user_role", "STRING",
                                 description="Role of the user"),
            bigquery.SchemaField("patient_age_bucket", "STRING",
                                 description="Generalized patient age (e.g., 75-80)"),
            bigquery.SchemaField("risk_score", "FLOAT64",
                                 description="Predicted mortality risk"),
            bigquery.SchemaField("risk_level", "STRING",
                                 description="HIGH, MODERATE, or LOW"),
            bigquery.SchemaField("target_hospital", "STRING",
                                 description="Target hospital for queries/transfers"),
            bigquery.SchemaField("query_type", "STRING",
                                 description="Type of query performed"),
            bigquery.SchemaField("data_accessed", "STRING",
                                 description="Data tables/resources accessed"),
            bigquery.SchemaField("tables_queried", "STRING", mode="REPEATED",
                                 description="List of tables queried"),
            bigquery.SchemaField("privacy_level", "STRING",
                                 description="Privacy protection level"),
            bigquery.SchemaField("k_anonymity_threshold", "INTEGER",
                                 description="K-anonymity threshold used"),
            bigquery.SchemaField("differential_privacy_epsilon", "FLOAT64",
                                 description="DP epsilon value if applied"),
            bigquery.SchemaField("hipaa_compliant", "BOOLEAN",
                                 description="Whether action was HIPAA compliant"),
            bigquery.SchemaField("legal_basis", "STRING",
                                 description="Legal basis for data access"),
            bigquery.SchemaField("purpose", "STRING",
                                 description="Purpose of the action"),
            bigquery.SchemaField("details", "JSON",
                                 description="Additional details as JSON"),
            bigquery.SchemaField("client_ip", "STRING",
                                 description="Client IP address"),
            bigquery.SchemaField("session_id", "STRING",
                                 description="Session identifier"),
        ]

        table = bigquery.Table(table_ref, schema=schema)
        table.description = f"HIMAS Audit Log for {HOSPITAL_ID} - HIPAA Compliance"
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="event_timestamp"
        )

        client.create_table(table)
        logger.info(f"✓ Created audit table: {table_ref}")
        _table_verified = True
        return True

    except Exception as e:
        logger.error(f"Failed to ensure audit table: {e}")
        return False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _generate_log_id(prefix: str = "log") -> str:
    """Generate unique log ID."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _get_risk_level(risk_score: float) -> str:
    """Determine risk level from score."""
    if risk_score > 0.7:
        return "HIGH"
    elif risk_score > 0.3:
        return "MODERATE"
    return "LOW"


def _safe_json(obj: Any) -> str:
    """Safely convert object to JSON string."""
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return json.dumps({"error": "serialization_failed"})


def _log_to_cloud_logging_struct(
    logger_name: str,
    log_entry: Dict[str, Any],
    severity: str = "INFO"
) -> bool:
    """
    Log structured data to Cloud Logging.

    Returns:
        True if successful, False otherwise
    """
    if not ENABLE_CLOUD_LOGGING:
        logger.debug(
            f"Cloud Logging disabled, struct log skipped: {logger_name}")
        return False

    client = _get_cloud_logging_client()
    if client is None:
        logger.warning("Cloud Logging client not available")
        return False

    try:
        cloud_logger = client.logger(logger_name)
        cloud_logger.log_struct(log_entry, severity=severity)
        logger.debug(f"✓ Struct logged to Cloud Logging: {logger_name}")
        return True
    except Exception as e:
        logger.error(f"Cloud Logging struct failed: {e}")
        return False


# ============================================================================
# MAIN AUDIT LOGGING FUNCTIONS
# ============================================================================

def log_prediction_access(
    patient_age_bucket: str,
    risk_score: float,
    user_id: str,
    action: str = "mortality_prediction",
    user_role: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Logs mortality prediction access to Cloud Logging and BigQuery.

    Args:
        patient_age_bucket: Generalized age (e.g., "75-80")
        risk_score: Predicted mortality risk
        user_id: Clinician who requested prediction
        action: Type of action performed
        user_role: Role of the user
        session_id: Session identifier

    Returns:
        Log confirmation with IDs
    """
    log_id = _generate_log_id("pred")
    event_timestamp = datetime.now()
    risk_level = _get_risk_level(risk_score)

    # Standard logging (goes to Cloud Logging via handler)
    logger.info(
        f"[AUDIT] Prediction access: {action} for age_bucket={patient_age_bucket}, risk={risk_level}")

    try:
        # Structured log entry
        log_entry = {
            "log_id": log_id,
            "hospital_id": HOSPITAL_ID,
            "action": action,
            "action_category": "prediction",
            "timestamp": event_timestamp.isoformat(),
            "user_id": user_id,
            "user_role": user_role,
            "patient_age_bucket": patient_age_bucket,
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "data_accessed": f"{HOSPITAL_ID}_data",
            "tables_queried": [f"{HOSPITAL_ID}_data"],
            "privacy_level": "k_anonymity_5",
            "k_anonymity_threshold": 5,
            "hipaa_compliant": True
        }

        # Log to Cloud Logging (structured)
        _log_to_cloud_logging_struct(
            f"himas-{HOSPITAL_ID}-audit",
            log_entry,
            severity="INFO"
        )

        # Log to BigQuery audit table
        _log_to_bigquery_audit(
            log_id=log_id,
            action=action,
            action_category="prediction",
            event_timestamp=event_timestamp,
            user_id=user_id,
            user_role=user_role,
            patient_age_bucket=patient_age_bucket,
            risk_score=risk_score,
            risk_level=risk_level,
            data_accessed=f"{HOSPITAL_ID}_data",
            tables_queried=[f"{HOSPITAL_ID}_data"],
            privacy_level="k_anonymity_5",
            k_anonymity_threshold=5,
            details=log_entry,
            session_id=session_id
        )

        logger.info(f"✓ Prediction access logged: {log_id}")

        return {
            "logged": True,
            "log_id": log_id,
            "risk_level": risk_level,
            "cloud_logging": ENABLE_CLOUD_LOGGING,
            "bigquery": _table_verified,
            "audit_trail": "Logged to Cloud Logging and BigQuery"
        }

    except Exception as e:
        logger.error(f"Audit logging failed: {str(e)}", exc_info=True)
        return {"logged": False, "error": str(e), "log_id": log_id}


def log_external_query(
    target_hospital: str,
    query_type: str,
    patient_anonymized: Dict[str, Any],
    user_id: str,
    user_role: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Logs cross-hospital queries for HIPAA audit trail.
    **CRITICAL for Compliance:** All external data sharing must be logged.

    Args:
        target_hospital: Hospital being queried
        query_type: Type of query (capability_check, similar_cases, etc.)
        patient_anonymized: Anonymized patient data sent
        user_id: Clinician who authorized query
        user_role: Role of the user
        session_id: Session identifier

    Returns:
        Log confirmation with audit ID
    """
    log_id = _generate_log_id("ext")
    event_timestamp = datetime.now()

    # Standard logging (goes to Cloud Logging via handler)
    logger.warning(
        f"[AUDIT] External query: {HOSPITAL_ID} -> {target_hospital}, type={query_type}")

    try:
        # ====================================================================
        # EXTRACT AND COMPUTE VALUES FROM patient_anonymized
        # ====================================================================

        # Get age bucket
        patient_age_bucket = patient_anonymized.get("age_bucket")

        # Extract risk_score - handle both numeric and range format
        risk_score = patient_anonymized.get("risk_score")
        if risk_score is None:
            # Try to extract from risk_score_range (e.g., "0.7-0.8")
            risk_range = patient_anonymized.get("risk_score_range", "")
            if risk_range and "-" in str(risk_range):
                try:
                    # Take the lower bound of the range
                    risk_score = float(str(risk_range).split("-")[0])
                except (ValueError, IndexError):
                    risk_score = None

        # Calculate risk level from score
        risk_level = None
        if risk_score is not None:
            risk_level = _get_risk_level(float(risk_score))

        # Build data_accessed string - what fields were shared
        shared_fields = []
        for key in ["age_bucket", "risk_score_range", "admission_type", "early_icu_score", "icu_type"]:
            if patient_anonymized.get(key):
                shared_fields.append(key)
        data_accessed = ", ".join(
            shared_fields) if shared_fields else "anonymized_patient_profile"

        # Tables queried - logical representation
        tables_queried = ["federated_hospital_data",
                          f"{target_hospital}_resources"]

        # Default user_role if not provided
        effective_user_role = user_role or patient_anonymized.get(
            "user_role") or "clinical_staff"

        # ====================================================================
        # BUILD LOG ENTRY
        # ====================================================================

        log_entry = {
            "log_id": log_id,
            "hospital_id": HOSPITAL_ID,
            "action": "external_query",
            "action_category": "external_query",
            "target_hospital": target_hospital,
            "query_type": query_type,
            "timestamp": event_timestamp.isoformat(),
            "user_id": user_id,
            "user_role": effective_user_role,
            "patient_age_bucket": patient_age_bucket,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "data_accessed": data_accessed,
            "tables_queried": tables_queried,
            "data_shared": {
                "age_bucket": patient_age_bucket,
                "risk_score_range": patient_anonymized.get("risk_score_range"),
                "admission_type": patient_anonymized.get("admission_type"),
                "early_icu_score": patient_anonymized.get("early_icu_score"),
                "required_capability": patient_anonymized.get("required_capabilities", [])
            },
            "privacy_level": "k_anonymity_5_dp_0.1",
            "k_anonymity_threshold": 5,
            "differential_privacy_epsilon": 0.1,
            "hipaa_compliant": True,
            "purpose": "Treatment - finding appropriate care facility",
            "legal_basis": "45 CFR 164.506(c)"
        }

        # Log to Cloud Logging (structured) - WARNING severity for external queries
        _log_to_cloud_logging_struct(
            f"himas-{HOSPITAL_ID}-external-queries",
            log_entry,
            severity="WARNING"
        )

        # Log to BigQuery audit table
        _log_to_bigquery_audit(
            log_id=log_id,
            action="external_query",
            action_category="external_query",
            event_timestamp=event_timestamp,
            user_id=user_id,
            user_role=effective_user_role,
            patient_age_bucket=patient_age_bucket,
            risk_score=risk_score,
            risk_level=risk_level,
            target_hospital=target_hospital,
            query_type=query_type,
            data_accessed=data_accessed,
            tables_queried=tables_queried,
            privacy_level="k_anonymity_5_dp_0.1",
            k_anonymity_threshold=5,
            differential_privacy_epsilon=0.1,
            purpose="Treatment - finding appropriate care facility",
            legal_basis="45 CFR 164.506(c)",
            hipaa_compliant=True,
            details=log_entry,
            session_id=session_id
        )

        logger.info(f"✓ External query logged: {log_id}")

        return {
            "logged": True,
            "log_id": log_id,
            "audit_id": log_id,
            "cloud_logging": ENABLE_CLOUD_LOGGING,
            "bigquery": _table_verified,
            "compliance_status": "HIPAA compliant - audit trail created"
        }

    except Exception as e:
        logger.error(f"External query logging failed: {str(e)}", exc_info=True)
        # CRITICAL: If we can't log, we shouldn't allow the query
        raise RuntimeError(
            f"Cannot proceed with external query - audit logging failed: {str(e)}"
        )


def log_transfer_event(
    target_hospital: str,
    patient_fingerprint: str,
    transfer_reason: str,
    urgency: str,
    user_id: str,
    patient_age_bucket: Optional[str] = None,
    user_role: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Logs patient transfer events for HIPAA compliance.

    Args:
        target_hospital: Hospital receiving the patient
        patient_fingerprint: Anonymized patient identifier (SHA-256 hash)
        transfer_reason: Clinical reason for transfer
        urgency: Transfer urgency (HIGH, MEDIUM, LOW)
        user_id: Clinician who initiated transfer
        patient_age_bucket: Generalized patient age
        user_role: Role of the user
        session_id: Session identifier

    Returns:
        Log confirmation with transfer ID
    """
    log_id = _generate_log_id("xfer")
    event_timestamp = datetime.now()

    # Standard logging (goes to Cloud Logging via handler)
    logger.warning(
        f"[AUDIT] Patient transfer: {HOSPITAL_ID} -> {target_hospital}, urgency={urgency}")

    try:
        log_entry = {
            "log_id": log_id,
            "hospital_id": HOSPITAL_ID,
            "action": "patient_transfer",
            "action_category": "transfer",
            "source_hospital": HOSPITAL_ID,
            "target_hospital": target_hospital,
            "timestamp": event_timestamp.isoformat(),
            "user_id": user_id,
            "user_role": user_role,
            "patient_age_bucket": patient_age_bucket,
            "patient_fingerprint": patient_fingerprint[:8] + "..." if patient_fingerprint else None,
            "purpose": transfer_reason,
            "urgency": urgency,
            "hipaa_compliant": True
        }

        # Log to Cloud Logging (structured) - WARNING severity for transfers
        _log_to_cloud_logging_struct(
            f"himas-{HOSPITAL_ID}-transfers",
            log_entry,
            severity="WARNING"
        )

        # Log to BigQuery audit table
        _log_to_bigquery_audit(
            log_id=log_id,
            action="patient_transfer",
            action_category="transfer",
            event_timestamp=event_timestamp,
            user_id=user_id,
            user_role=user_role,
            patient_age_bucket=patient_age_bucket,
            target_hospital=target_hospital,
            purpose=transfer_reason,
            details=log_entry,
            session_id=session_id
        )

        logger.info(f"✓ Transfer logged: {log_id}")

        return {
            "logged": True,
            "log_id": log_id,
            "transfer_id": log_id,
            "cloud_logging": ENABLE_CLOUD_LOGGING,
            "bigquery": _table_verified,
            "compliance_status": "HIPAA compliant - transfer audit trail created"
        }

    except Exception as e:
        logger.error(f"Transfer logging failed: {str(e)}", exc_info=True)
        raise RuntimeError(
            f"Cannot proceed with transfer - audit logging failed: {str(e)}"
        )


def log_transfer_receipt(
    transfer_id: str,
    source_hospital: str,
    receiving_hospital: str,
    patient_anonymized: Dict[str, Any],
    transfer_reason: str,
    urgency: str,
    bed_reservation: Dict[str, Any],
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Log transfer receipt at the RECEIVING hospital's audit table.

    When a patient is transferred TO a hospital, that hospital's audit
    table should also have a record of the incoming transfer.

    Args:
        transfer_id: Transfer identifier
        source_hospital: Hospital sending the patient
        receiving_hospital: Hospital receiving the patient (whose table to log to)
        patient_anonymized: Anonymized patient data
        transfer_reason: Reason for transfer
        urgency: Transfer urgency
        bed_reservation: Bed reservation details
        user_id: User initiating (default: system)

    Returns:
        dict with logging result
    """
    from google.cloud import bigquery
    log_id = f"recv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    event_timestamp = datetime.now()

    logger.info(
        f"[AUDIT] Logging transfer receipt at {receiving_hospital}: {transfer_id}")

    # Extract patient info
    patient_age_bucket = patient_anonymized.get("age_bucket", "unknown")

    # Extract risk_score
    risk_score = patient_anonymized.get("risk_score")
    if risk_score is None:
        risk_range = patient_anonymized.get("risk_score_range", "")
        if risk_range and "-" in str(risk_range):
            try:
                risk_score = float(str(risk_range).split("-")[0])
            except (ValueError, IndexError):
                risk_score = None

    risk_level = _get_risk_level(
        float(risk_score)) if risk_score is not None else None

    # Build log entry
    log_entry = {
        "log_id": log_id,
        "transfer_id": transfer_id,
        "action": "transfer_receipt",
        "action_category": "incoming_transfer",
        "hospital_id": receiving_hospital,
        "source_hospital": source_hospital,
        "timestamp": event_timestamp.isoformat(),
        "user_id": user_id,
        "patient_age_bucket": patient_age_bucket,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "transfer_reason": transfer_reason,
        "urgency": urgency,
        "bed_reservation": bed_reservation,
        "data_received": {
            "age_bucket": patient_age_bucket,
            "admission_type": patient_anonymized.get("admission_type"),
            "early_icu_score": patient_anonymized.get("early_icu_score"),
            "risk_score_range": patient_anonymized.get("risk_score_range"),
            "icu_type": patient_anonymized.get("icu_type")
        },
        "privacy_level": "k_anonymity_5_dp_0.1",
        "hipaa_compliant": True
    }

    result = {
        "log_id": log_id,
        "transfer_id": transfer_id,
        "receiving_hospital": receiving_hospital,
        "source_hospital": source_hospital,
        "console": True,
        "bigquery": False
    }

    # Log to BigQuery at RECEIVING hospital's table
    try:
        # Determine receiving hospital's audit table
        receiving_table = f"{GOOGLE_CLOUD_PROJECT}.{AUDIT_LOG_DATASET}.{receiving_hospital}_audit_log"

        client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT)

        # Build row for BigQuery
        row = {
            "log_id": log_id,
            "hospital_id": receiving_hospital,
            "action": "transfer_receipt",
            "action_category": "incoming_transfer",
            "event_timestamp": event_timestamp.isoformat(),
            "logged_at": datetime.now().isoformat(),
            "user_id": user_id,
            "user_role": "receiving_coordinator",
            "patient_age_bucket": patient_age_bucket,
            "risk_score": risk_score,
            "risk_level": risk_level,
            # In this context, "target" is where it came FROM
            "target_hospital": source_hospital,
            "query_type": "incoming_transfer",
            "data_accessed": f"transfer_id: {transfer_id}, reason: {transfer_reason}",
            "tables_queried": ["transfer_coordination", f"{source_hospital}_outbound"],
            "privacy_level": "k_anonymity_5_dp_0.1",
            "k_anonymity_threshold": 5,
            "differential_privacy_epsilon": 0.1,
            "hipaa_compliant": True,
            "legal_basis": "45 CFR 164.506(c) - Treatment",
            "purpose": f"Incoming transfer from {source_hospital} for {transfer_reason}",
            "details": json.dumps(log_entry, default=str),
            "session_id": transfer_id
        }

        errors = client.insert_rows_json(receiving_table, [row])

        if errors:
            logger.error(
                f"BigQuery insert errors at {receiving_hospital}: {errors}")
            result["bigquery_error"] = str(errors)
        else:
            result["bigquery"] = True
            result["bigquery_table"] = receiving_table
            logger.info(
                f"✓ Transfer receipt logged to {receiving_table}: {log_id}")

    except Exception as e:
        logger.error(f"Failed to log transfer receipt to BigQuery: {e}")
        result["bigquery_error"] = str(e)

    return result


def log_privacy_operation(
    operation: str,
    source_hospital: str,
    target_hospital: str,
    fields_shared: int,
    pii_removed: int,
    user_id: str = "system",
    k_anonymity_met: bool = True,
    differential_privacy_applied: bool = False,
    epsilon: Optional[float] = None,
    patient_fingerprint: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Log privacy-specific operations (anonymization, data sharing).

    This is called by privacy_tools.py for tracking privacy operations.

    Args:
        operation: Operation name (e.g., "peer_hospital_query", "transfer_initiation")
        source_hospital: Source hospital ID
        target_hospital: Target hospital or coordinator
        fields_shared: Number of fields shared
        pii_removed: Number of PII fields removed
        user_id: User who initiated the operation
        k_anonymity_met: Whether k-anonymity was satisfied
        differential_privacy_applied: Whether DP was applied
        epsilon: DP epsilon value
        patient_fingerprint: Anonymized patient ID
        request_id: Request correlation ID
        session_id: Session ID

    Returns:
        Log confirmation with audit ID
    """
    log_id = _generate_log_id("priv")
    event_timestamp = datetime.now()

    # Standard logging
    logger.info(
        f"[AUDIT] Privacy operation: {operation} | {source_hospital} -> {target_hospital}")
    logger.info(
        f"  Fields shared: {fields_shared}, PII removed: {pii_removed}")

    try:
        log_entry = {
            "log_id": log_id,
            "hospital_id": source_hospital,
            "action": operation,
            "action_category": "privacy",
            "source_hospital": source_hospital,
            "target_hospital": target_hospital,
            "timestamp": event_timestamp.isoformat(),
            "user_id": user_id,
            "fields_shared": fields_shared,
            "pii_removed": pii_removed,
            "k_anonymity_met": k_anonymity_met,
            "differential_privacy_applied": differential_privacy_applied,
            "differential_privacy_epsilon": epsilon,
            "patient_fingerprint": patient_fingerprint[:8] + "..." if patient_fingerprint else None,
            "request_id": request_id,
            "privacy_level": f"k_anonymity_5{'_dp_' + str(epsilon) if differential_privacy_applied else ''}",
            "hipaa_compliant": True
        }

        # Log to Cloud Logging (structured)
        _log_to_cloud_logging_struct(
            f"himas-{source_hospital}-privacy",
            log_entry,
            severity="INFO"
        )

        # Log to BigQuery
        _log_to_bigquery_audit(
            log_id=log_id,
            action=operation,
            action_category="privacy",
            event_timestamp=event_timestamp,
            user_id=user_id,
            target_hospital=target_hospital,
            privacy_level=log_entry["privacy_level"],
            k_anonymity_threshold=5,
            differential_privacy_epsilon=epsilon,
            hipaa_compliant=True,
            details=log_entry,
            session_id=session_id
        )

        logger.info(f"✓ Privacy operation logged: {log_id}")

        return {
            "logged": True,
            "log_id": log_id,
            "audit_id": log_id,
            "cloud_logging": ENABLE_CLOUD_LOGGING,
            "bigquery": _table_verified
        }

    except Exception as e:
        logger.error(
            f"Privacy operation logging failed: {str(e)}", exc_info=True)
        return {"logged": False, "error": str(e), "log_id": log_id}


def get_audit_trail(
    hours: int = 24,
    action_filter: Optional[str] = None,
    user_id: Optional[str] = None
) -> dict[str, Any]:
    """
    Retrieve audit trail entries for compliance review.

    Args:
        hours: Number of hours to look back (default: 24)
        action_filter: Optional filter for specific action types
                      (e.g., 'risk_prediction', 'transfer_initiation', 'similar_case_query')
        user_id: Optional filter for specific user

    Returns:
        dict containing audit trail entries (with PII removed)
    """
    from google.cloud import bigquery
    from datetime import datetime, timedelta, timezone

    try:
        client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT)

        # Calculate cutoff time with explicit UTC timezone
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_str = cutoff_time.strftime('%Y-%m-%dT%H:%M:%S')

        # Query using actual schema column names
        query = f"""
        SELECT 
            log_id,
            hospital_id,
            action,
            action_category,
            event_timestamp,
            logged_at,
            user_id,
            user_role,
            patient_age_bucket,
            risk_score,
            risk_level,
            target_hospital,
            query_type,
            data_accessed,
            tables_queried,
            privacy_level,
            k_anonymity_threshold,
            differential_privacy_epsilon,
            hipaa_compliant,
            legal_basis,
            purpose,
            details
        FROM `{GOOGLE_CLOUD_PROJECT}.{AUDIT_LOG_DATASET}.{AUDIT_LOG_TABLE}`
        WHERE logged_at >= TIMESTAMP('{cutoff_str}')
        """

        if action_filter:
            safe_action = action_filter.replace("'", "''")
            query += f" AND query_type = '{safe_action}'"

        if user_id:
            safe_user = user_id.replace("'", "''")
            query += f" AND user_id = '{safe_user}'"

        query += " ORDER BY logged_at DESC LIMIT 100"

        logger.info(
            f"Querying audit trail: last {hours} hours, query_type={action_filter}")

        result = client.query(query).to_dataframe()

        if result.empty:
            return {
                "entries": [],
                "count": 0,
                "time_range_hours": hours,
                "filters_applied": {
                    "action": action_filter,
                    "user_id": user_id
                },
                "message": "No audit entries found for the specified criteria"
            }

        # Convert to list of dicts
        entries = []
        for _, row in result.iterrows():
            # Handle timestamps
            logged_at_val = row.get("logged_at")
            event_ts_val = row.get("event_timestamp")

            entry = {
                "log_id": row.get("log_id"),
                "hospital_id": row.get("hospital_id"),
                "action": row.get("action"),
                "action_category": row.get("action_category"),
                "event_timestamp": event_ts_val.isoformat() if hasattr(event_ts_val, 'isoformat') else str(event_ts_val),
                "logged_at": logged_at_val.isoformat() if hasattr(logged_at_val, 'isoformat') else str(logged_at_val),
                "user_id": row.get("user_id"),
                "user_role": row.get("user_role"),
                "patient_age_bucket": row.get("patient_age_bucket"),
                "risk_score": float(row.get("risk_score")) if row.get("risk_score") is not None else None,
                "risk_level": row.get("risk_level"),
                "target_hospital": row.get("target_hospital"),
                "query_type": row.get("query_type"),
                "data_accessed": row.get("data_accessed"),
                "tables_queried": list(row.get("tables_queried")) if row.get("tables_queried") is not None else [],
                "privacy_level": row.get("privacy_level"),
                "k_anonymity_threshold": int(row.get("k_anonymity_threshold")) if row.get("k_anonymity_threshold") is not None else None,
                "differential_privacy_epsilon": float(row.get("differential_privacy_epsilon")) if row.get("differential_privacy_epsilon") is not None else None,
                "hipaa_compliant": bool(row.get("hipaa_compliant")),
                "legal_basis": row.get("legal_basis"),
                "purpose": row.get("purpose")
            }
            entries.append(entry)

        return {
            "entries": entries,
            "count": len(entries),
            "time_range_hours": hours,
            "filters_applied": {
                "query_type": action_filter,
                "user_id": user_id
            },
            "query_timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Audit trail retrieval failed: {e}")
        return {
            "error": str(e),
            "entries": [],
            "count": 0
        }


def get_logging_status() -> Dict[str, Any]:
    """
    Get status of logging infrastructure.

    Returns:
        Status of Cloud Logging and BigQuery connections
    """
    return {
        "cloud_logging": {
            "enabled": ENABLE_CLOUD_LOGGING,
            "initialized": _cloud_logging_initialized,
            "client_available": _cloud_logging_client is not None
        },
        "bigquery": {
            "client_available": _bigquery_client is not None,
            "table_verified": _table_verified,
            "table_name": f"{GOOGLE_CLOUD_PROJECT}.{AUDIT_LOG_DATASET}.{AUDIT_LOG_TABLE}"
        },
        "hospital_id": HOSPITAL_ID,
        "project_id": GOOGLE_CLOUD_PROJECT
    }


# ============================================================================
# BIGQUERY LOGGING HELPER
# ============================================================================

def _log_to_bigquery_audit(
    log_id: str,
    action: str,
    event_timestamp: datetime,
    user_id: str,
    action_category: Optional[str] = None,
    user_role: Optional[str] = None,
    patient_age_bucket: Optional[str] = None,
    risk_score: Optional[float] = None,
    risk_level: Optional[str] = None,
    target_hospital: Optional[str] = None,
    query_type: Optional[str] = None,
    data_accessed: Optional[str] = None,
    tables_queried: Optional[List[str]] = None,
    privacy_level: Optional[str] = None,
    k_anonymity_threshold: Optional[int] = None,
    differential_privacy_epsilon: Optional[float] = None,
    hipaa_compliant: bool = True,
    legal_basis: Optional[str] = None,
    purpose: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    client_ip: Optional[str] = None,
    session_id: Optional[str] = None
) -> bool:
    """
    Logs to BigQuery hospital audit table.

    Uses table: audit_logs.{hospital_id}_audit_log

    Returns:
        True if successful, False otherwise
    """
    # Ensure table exists first
    if not ensure_audit_table_exists():
        logger.warning(
            f"BigQuery table not available, audit logged to Cloud Logging only: {log_id}")
        return False

    client = _get_bigquery_client()
    if client is None:
        logger.warning("BigQuery client not available")
        return False

    try:
        table_id = f"{GOOGLE_CLOUD_PROJECT}.{AUDIT_LOG_DATASET}.{AUDIT_LOG_TABLE}"

        row = {
            "log_id": str(log_id),
            "hospital_id": str(HOSPITAL_ID),
            "action": str(action),
            "action_category": str(action_category) if action_category else None,
            "event_timestamp": event_timestamp.isoformat(),
            "logged_at": datetime.now().isoformat(),
            "user_id": str(user_id),
            "user_role": str(user_role) if user_role else None,
            "patient_age_bucket": str(patient_age_bucket) if patient_age_bucket else None,
            "risk_score": float(risk_score) if risk_score is not None else None,
            "risk_level": str(risk_level) if risk_level else None,
            "target_hospital": str(target_hospital) if target_hospital else None,
            "query_type": str(query_type) if query_type else None,
            "data_accessed": str(data_accessed) if data_accessed else None,
            "tables_queried": [str(t) for t in tables_queried] if tables_queried else None,
            "privacy_level": str(privacy_level) if privacy_level else None,
            "k_anonymity_threshold": int(k_anonymity_threshold) if k_anonymity_threshold else None,
            "differential_privacy_epsilon": float(differential_privacy_epsilon) if differential_privacy_epsilon else None,
            "hipaa_compliant": bool(hipaa_compliant),
            "legal_basis": str(legal_basis) if legal_basis else None,
            "purpose": str(purpose) if purpose else None,
            "details": _safe_json(details) if details else None,
            "client_ip": str(client_ip) if client_ip else None,
            "session_id": str(session_id) if session_id else None
        }

        # Remove None values for cleaner insert
        row = {k: v for k, v in row.items() if v is not None}

        errors = client.insert_rows_json(table_id, [row])

        if errors:
            logger.error(f"BigQuery audit insert errors: {errors}")
            return False
        else:
            logger.debug(f"✓ Logged to BigQuery: {table_id}")
            return True

    except Exception as e:
        logger.error(f"BigQuery audit logging failed: {str(e)}", exc_info=True)
        return False


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Try to ensure audit table exists on module load
# This will create the table if it doesn't exist
if ENABLE_CLOUD_LOGGING:
    try:
        ensure_audit_table_exists()
    except Exception as e:
        logger.warning(f"Could not verify audit table on startup: {e}")
