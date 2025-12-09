"""
Mortality Risk Prediction Tool (Uses DataPreprocessor)
Automatically logs all predictions to BigQuery audit table.
"""
import numpy as np
import logging
import pandas as pd
import tensorflow as tf
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from .model_loader import get_model_loader
from ....config import config
from ....utils.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default values for ALL possible features required by the model
DEFAULT_PATIENT_FEATURES = {
    # Demographics
    'age_at_admission': 65,
    'gender': 'M',
    'race': 'UNKNOWN',
    'marital_status': 'UNKNOWN',
    'insurance': 'Other',
    # Admission context
    'admission_type': 'EMERGENCY',
    'admission_location': 'EMERGENCY ROOM',
    'hours_admit_to_icu': 24.0,
    'early_icu_score': 1,
    # ICU assignment
    'icu_type': 'Medical ICU',
    'first_careunit': 'Medical Intensive Care Unit (MICU)',
    # Timing flags
    'weekend_admission': 0,
    'night_admission': 0,
    'ed_admission_flag': 0,
    'emergency_admission_flag': 0,
    # Hospital LOS
    'los_hospital_days': 0,
    'los_hospital_hours': 0,
    # Temporal features (DEFAULT = 0 for admission-time)
    'los_icu_hours': 0.0,
    'los_icu_days': 0.0,
    'n_icu_transfers': 0,
    'n_total_transfers': 0,
    'n_distinct_icu_units': 1,
    'is_mixed_icu': 0,
    # Identifiers (excluded by preprocessor)
    'stay_id': 0,
    'subject_id': 0,
    'hadm_id': 0,
    # Target (ignored)
    'icu_mortality_label': 0
}


# ============================================================================
# AUDIT LOGGING HELPER
# ============================================================================

def _auto_log_prediction_to_bigquery(
    patient_age_bucket: str,
    risk_score: float,
    risk_level: str,
    prediction_mode: str,
    early_icu_score: int,
    admission_type: str
) -> Optional[str]:
    """
    Automatically log a prediction to BigQuery audit table.

    Args:
        patient_age_bucket: Privacy-preserving age bucket (e.g., "75-80")
        risk_score: Predicted mortality risk (0-1)
        risk_level: Categorized risk (LOW/MODERATE/HIGH)
        prediction_mode: ADMISSION-TIME or UPDATED
        early_icu_score: Early ICU admission score (0-3)
        admission_type: Type of admission

    Returns:
        log_id if successful, None if failed
    """
    try:
        from ...privacy_guardian.tools.audit_logging import (
            _log_to_bigquery_audit,
            HOSPITAL_ID
        )

        log_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Build details JSON
        details = {
            "action": "mortality_prediction",
            "action_category": "prediction",
            "patient_age_bucket": patient_age_bucket,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "prediction_mode": prediction_mode,
            "early_icu_score": early_icu_score,
            "admission_type": admission_type,
            "hospital_id": HOSPITAL_ID,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        success = _log_to_bigquery_audit(
            log_id=log_id,
            action="mortality_prediction",
            action_category="prediction",
            event_timestamp=datetime.now(timezone.utc),
            user_id="clinical_decision_agent",
            hipaa_compliant=True,
            user_role="clinical_staff",
            patient_age_bucket=patient_age_bucket,
            risk_score=float(risk_score),
            risk_level=risk_level,
            data_accessed=f"{HOSPITAL_ID}_data",
            tables_queried=[f"{HOSPITAL_ID}_data"],
            privacy_level="k_anonymity_5",
            k_anonymity_threshold=5,
            purpose=f"ICU mortality risk assessment ({prediction_mode})",
            details=details
        )

        if success:
            logger.info(f"Prediction auto-logged to BigQuery: {log_id}")
            return log_id
        else:
            logger.warning(f"Prediction logging returned False: {log_id}")
            return None

    except ImportError as e:
        logger.warning(f"Could not import audit logging module: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to auto-log prediction to BigQuery: {e}")
        return None


def _create_age_bucket(age: int) -> str:
    """Create privacy-preserving age bucket (5-year ranges)."""
    bucket_start = (age // 5) * 5
    return f"{bucket_start}-{bucket_start + 5}"


# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

def predict_mortality_risk(patient_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predicts ICU mortality risk using SAME preprocessing as training.
    Automatically logs prediction to BigQuery audit table.

    Args:
        patient_features: Dictionary with patient data

    Returns:
        Risk assessment dictionary including audit_log_id
    """
    try:
        # Load model AND preprocessor (cached)
        loader = get_model_loader()
        model, preprocessor = loader.load_all()
        logger.info("Model and preprocessor loaded successfully")

        # Prepare data
        patient_df = _create_patient_dataframe(patient_features)
        logger.info(f"Patient DataFrame created: {patient_df.shape}")

        # Transform using FITTED preprocessor
        X_processed, _ = preprocessor.transform(
            patient_df,
            numerical_features=config.NUMERICAL_FEATURES,
            categorical_features=config.CATEGORICAL_FEATURES,
            target=config.TARGET_COLUMN,
            excluded=config.EXCLUDED_COLUMNS
        )
        logger.info(f"Patient data preprocessed: shape={X_processed.shape}")

        # Make prediction
        logger.info("Starting model inference...")
        X_tensor = tf.constant(X_processed, dtype=tf.float32)

        with tf.device('/CPU:0'):
            prediction_tensor = model(X_tensor, training=False)
            prediction = prediction_tensor.numpy()

        logger.info("Model inference complete")
        risk_score = float(prediction[0][0])

        # Validate and clamp
        if not 0 <= risk_score <= 1:
            logger.warning(
                f"Invalid risk score: {risk_score}. Clamping to [0,1]")
            risk_score = np.clip(risk_score, 0, 1)

        # Categorize
        risk_level = _categorize_risk(risk_score)

        # Get key factors
        key_factors = _get_key_factors(patient_features, risk_score)

        # Determine prediction mode
        temporal_features_provided = sum([
            patient_features.get('los_icu_hours', 0) > 0,
            patient_features.get('n_icu_transfers', 0) > 0
        ])
        prediction_mode = "UPDATED" if temporal_features_provided > 0 else "ADMISSION-TIME"

        # ====================================================================
        # AUTO-LOG TO BIGQUERY AUDIT TABLE
        # ====================================================================
        age = patient_features.get(
            'age_at_admission', DEFAULT_PATIENT_FEATURES['age_at_admission'])
        age_bucket = _create_age_bucket(age)
        early_icu_score = patient_df['early_icu_score'].iloc[0] if 'early_icu_score' in patient_df.columns else 1
        admission_type = patient_features.get('admission_type')

        audit_log_id = _auto_log_prediction_to_bigquery(
            patient_age_bucket=age_bucket,
            risk_score=risk_score,
            risk_level=risk_level,
            prediction_mode=prediction_mode,
            early_icu_score=int(early_icu_score),
            admission_type=admission_type
        )
        # ====================================================================

        result = {
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "risk_percentage": f"{round(risk_score * 100, 1)}%",
            "prediction_mode": prediction_mode,
            "key_factors": key_factors,
            "hospital_id": config.HOSPITAL_ID,
            "prediction_timestamp": datetime.now().isoformat(),
            "audit_log_id": audit_log_id,
            "audit_logged": audit_log_id is not None,
            "preprocessing": {
                "method": "DataPreprocessor (fitted on BigQuery training data)",
                "feature_dim": preprocessor.feature_dim,
                "same_as_training": True
            }
        }

        logger.info(
            f"Prediction: Risk={risk_score:.3f} ({risk_level}) - Mode: {prediction_mode} - Audit: {audit_log_id}"
        )

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.exception("Full traceback:")
        raise RuntimeError(f"Prediction error: {str(e)}")


def _create_patient_dataframe(patient_features: Dict[str, Any]) -> pd.DataFrame:
    """Creates DataFrame from patient features with ALL required columns."""
    # Start with defaults
    complete_features = DEFAULT_PATIENT_FEATURES.copy()

    # Update with provided features
    complete_features.update(patient_features)

    # Calculate derived fields if timestamps provided
    if 'admittime' in patient_features:
        if 'weekend_admission' not in patient_features:
            complete_features['weekend_admission'] = FeatureExtractor.is_weekend(
                patient_features['admittime'])
        if 'night_admission' not in patient_features:
            complete_features['night_admission'] = FeatureExtractor.is_night_admission(
                patient_features['admittime'])

    # Derive flags
    if 'admission_location' in patient_features and 'ed_admission_flag' not in patient_features:
        complete_features['ed_admission_flag'] = 1 if 'EMERGENCY' in str(
            patient_features['admission_location']).upper() else 0

    if 'admission_type' in patient_features and 'emergency_admission_flag' not in patient_features:
        complete_features['emergency_admission_flag'] = 1 if patient_features['admission_type'] == 'EMERGENCY' else 0

    # ALWAYS calculate early_icu_score from hours_admit_to_icu (source of truth)
    if 'hours_admit_to_icu' in patient_features:
        hours = patient_features['hours_admit_to_icu']
        calculated_score = FeatureExtractor.calculate_early_icu_score(hours)

        if 'early_icu_score' in patient_features:
            provided_score = patient_features['early_icu_score']
            if provided_score != calculated_score:
                logger.warning(
                    f"early_icu_score mismatch: provided={provided_score}, "
                    f"calculated={calculated_score} (from hours_admit_to_icu={hours}). "
                    f"Using calculated value."
                )
        complete_features['early_icu_score'] = calculated_score
        logger.info(
            f"early_icu_score calculated: {calculated_score} (hours_admit_to_icu={hours})")

    # Estimate hospital LOS if not provided
    if 'hours_admit_to_icu' in patient_features:
        if 'los_hospital_hours' not in patient_features:
            complete_features['los_hospital_hours'] = patient_features['hours_admit_to_icu']
        if 'los_hospital_days' not in patient_features:
            complete_features['los_hospital_days'] = patient_features['hours_admit_to_icu'] / 24.0

    return pd.DataFrame([complete_features])


def _categorize_risk(risk_score: float) -> str:
    """Categorizes risk score."""
    if risk_score < 0.3:
        return "LOW"
    elif risk_score < 0.7:
        return "MODERATE"
    else:
        return "HIGH"


def _get_key_factors(patient_features: Dict[str, Any], risk_score: float) -> List[Dict[str, Any]]:
    """Identifies top risk factors."""
    factors = []

    # Early ICU
    early_icu = patient_features.get('early_icu_score', 0)
    if early_icu >= 2:
        factors.append({
            "feature": "early_icu_score",
            "value": early_icu,
            "explanation": f"{'Very early' if early_icu == 3 else 'Early'} ICU admission indicates critical illness"
        })

    # Age
    age = patient_features.get('age_at_admission', 65)
    if age >= 75:
        factors.append({
            "feature": "age_at_admission",
            "value": age,
            "explanation": f"Advanced age ({age} years) increases baseline mortality risk"
        })

    # Emergency
    if patient_features.get('admission_type') == 'EMERGENCY':
        factors.append({
            "feature": "emergency_admission",
            "value": 1,
            "explanation": "Emergency admission suggests acute decompensation"
        })

    # Transfers (temporal)
    n_transfers = patient_features.get('n_icu_transfers', 0)
    if n_transfers > 0:
        factors.append({
            "feature": "n_icu_transfers",
            "value": n_transfers,
            "explanation": f"{n_transfers} ICU transfer(s) indicate complications"
        })

    # ICU LOS (temporal)
    los_icu_hours = patient_features.get('los_icu_hours', 0)
    if los_icu_hours > 48:
        factors.append({
            "feature": "los_icu_hours",
            "value": los_icu_hours,
            "explanation": f"Extended ICU stay ({los_icu_hours:.0f} hours) indicates severity"
        })

    return factors[:3]
