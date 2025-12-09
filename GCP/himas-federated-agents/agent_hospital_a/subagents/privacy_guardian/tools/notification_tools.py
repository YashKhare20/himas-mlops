"""
Notification Tools for HIMAS
Sends email alerts for critical events like patient transfers.
Uses Gmail SMTP via Google Cloud configuration.
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Email Configuration (from environment or defaults)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")  # Your Gmail address
# App password (not regular password)
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", SMTP_USER)
SENDER_NAME = os.getenv("SENDER_NAME", "HIMAS Transfer Alerts")

# Default recipients for transfer alerts (comma-separated in env var)
DEFAULT_TRANSFER_RECIPIENTS = os.getenv(
    "TRANSFER_ALERT_EMAILS",
    "yashkharess20@gmail.com"
).split(",")

# Hospital contact emails (would be from database in production)
HOSPITAL_CONTACTS = {
    "hospital_a": os.getenv("HOSPITAL_A_EMAIL"),
    "hospital_b": os.getenv("HOSPITAL_B_EMAIL"),
    "hospital_c": os.getenv("HOSPITAL_C_EMAIL"),
}


def send_transfer_notification(
    transfer_id: str,
    source_hospital: str,
    target_hospital: str,
    transfer_reason: str,
    urgency: str,
    patient_age_bucket: str,
    risk_level: str,
    bed_reservation: Dict[str, Any],
    estimated_transport_minutes: int,
    additional_recipients: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Send email notification for patient transfer.

    Notifies:
    - Default alert recipients (admin/monitoring)
    - Receiving hospital contact
    - Any additional specified recipients

    Args:
        transfer_id: Unique transfer identifier
        source_hospital: Hospital initiating transfer
        target_hospital: Hospital receiving patient
        transfer_reason: Clinical reason for transfer
        urgency: Transfer urgency (HIGH, MEDIUM, LOW)
        patient_age_bucket: Anonymized age range
        risk_level: Patient risk level (LOW, MODERATE, HIGH, CRITICAL)
        bed_reservation: Bed reservation details
        estimated_transport_minutes: ETA in minutes
        additional_recipients: Optional extra email addresses

    Returns:
        dict with notification status
    """
    notification_id = f"notif_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(
        f"[{notification_id}] Sending transfer notification: {source_hospital} → {target_hospital}")

    # Check if SMTP is configured
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning(
            "SMTP credentials not configured - skipping email notification")
        return {
            "notification_id": notification_id,
            "sent": False,
            "reason": "SMTP credentials not configured",
            "transfer_id": transfer_id
        }

    try:
        # Build recipient list
        recipients = list(DEFAULT_TRANSFER_RECIPIENTS)

        # Add receiving hospital contact
        if target_hospital in HOSPITAL_CONTACTS:
            recipients.append(HOSPITAL_CONTACTS[target_hospital])

        # Add additional recipients
        if additional_recipients:
            recipients.extend(additional_recipients)

        # Remove duplicates and empty strings
        recipients = list(set(r.strip() for r in recipients if r.strip()))

        if not recipients:
            logger.warning("No recipients configured - skipping notification")
            return {
                "notification_id": notification_id,
                "sent": False,
                "reason": "No recipients configured"
            }

        # Build email content
        subject = _build_transfer_subject(
            transfer_id, urgency, source_hospital, target_hospital)
        html_body = _build_transfer_email_html(
            transfer_id=transfer_id,
            source_hospital=source_hospital,
            target_hospital=target_hospital,
            transfer_reason=transfer_reason,
            urgency=urgency,
            patient_age_bucket=patient_age_bucket,
            risk_level=risk_level,
            bed_reservation=bed_reservation,
            estimated_transport_minutes=estimated_transport_minutes
        )

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
        msg["To"] = ", ".join(recipients)

        # Attach HTML body
        msg.attach(MIMEText(html_body, "html"))

        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())

        logger.info(
            f"✓ Transfer notification sent to {len(recipients)} recipients")

        return {
            "notification_id": notification_id,
            "sent": True,
            "transfer_id": transfer_id,
            "recipients_count": len(recipients),
            "recipients": recipients,
            "urgency": urgency,
            "timestamp": datetime.now().isoformat()
        }

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP authentication failed: {e}")
        return {
            "notification_id": notification_id,
            "sent": False,
            "reason": "SMTP authentication failed - check credentials",
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Failed to send transfer notification: {e}")
        return {
            "notification_id": notification_id,
            "sent": False,
            "reason": str(e)
        }


def _build_transfer_subject(
    transfer_id: str,
    urgency: str,
    source_hospital: str,
    target_hospital: str
) -> str:
    """Build email subject line."""
    urgency_emoji = {
        "HIGH": "🔴",
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }.get(urgency.upper(), "⚪")

    return f"{urgency_emoji} HIMAS Transfer Alert [{urgency}]: {source_hospital.upper()} → {target_hospital.upper()} | {transfer_id}"


def _build_transfer_email_html(
    transfer_id: str,
    source_hospital: str,
    target_hospital: str,
    transfer_reason: str,
    urgency: str,
    patient_age_bucket: str,
    risk_level: str,
    bed_reservation: Dict[str, Any],
    estimated_transport_minutes: int
) -> str:
    """Build HTML email body for transfer notification."""

    # Urgency styling
    urgency_colors = {
        "HIGH": "#dc3545",
        "MEDIUM": "#ffc107",
        "LOW": "#28a745"
    }
    urgency_color = urgency_colors.get(urgency.upper(), "#6c757d")

    # Risk level styling
    risk_colors = {
        "LOW": "#28a745",
        "MODERATE": "#ffc107",
        "HIGH": "#dc3545",
        "CRITICAL": "#721c24"
    }
    risk_color = risk_colors.get(risk_level.upper(), "#6c757d")

    # Hospital display names
    hospital_names = {
        "hospital_a": "Hospital A (Community Hospital)",
        "hospital_b": "Hospital B (Tertiary Care Center)",
        "hospital_c": "Hospital C (Rural Hospital)"
    }

    source_name = hospital_names.get(source_hospital, source_hospital)
    target_name = hospital_names.get(target_hospital, target_hospital)

    # Bed info
    bed_id = bed_reservation.get("bed_id", "TBD")
    bed_unit = bed_reservation.get("unit", "ICU")
    reserved_until = bed_reservation.get("reserved_until", "N/A")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; }}
            .container {{ max-width: 600px; margin: 0 auto; background: white; }}
            .header {{ background: linear-gradient(135deg, {urgency_color} 0%, #343a40 100%); color: white; padding: 30px; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 24px; }}
            .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
            .content {{ padding: 30px; }}
            .transfer-box {{ background: #f8f9fa; border-left: 5px solid {urgency_color}; padding: 20px; margin: 20px 0; border-radius: 5px; }}
            .hospital-flow {{ display: flex; justify-content: center; align-items: center; margin: 20px 0; }}
            .hospital {{ background: white; padding: 15px 25px; border-radius: 10px; border: 2px solid #dee2e6; text-align: center; }}
            .arrow {{ font-size: 30px; margin: 0 20px; color: {urgency_color}; }}
            .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }}
            .info-item {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
            .info-label {{ font-size: 12px; color: #6c757d; text-transform: uppercase; margin-bottom: 5px; }}
            .info-value {{ font-size: 18px; font-weight: bold; color: #343a40; }}
            .risk-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; background: {risk_color}; font-weight: bold; }}
            .urgency-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; background: {urgency_color}; font-weight: bold; }}
            .bed-info {{ background: #e7f3ff; border: 1px solid #0066cc; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .privacy-note {{ background: #d4edda; border: 1px solid #28a745; padding: 15px; border-radius: 5px; margin: 20px 0; font-size: 14px; }}
            .footer {{ background: #343a40; color: white; padding: 20px; text-align: center; }}
            .footer p {{ margin: 5px 0; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚑 PATIENT TRANSFER INITIATED</h1>
                <p>HIMAS Federated Healthcare Network</p>
            </div>
            
            <div class="content">
                <div class="transfer-box">
                    <div style="text-align: center; margin-bottom: 15px;">
                        <span class="urgency-badge">{urgency.upper()} URGENCY</span>
                    </div>
                    
                    <table style="width: 100%; text-align: center;">
                        <tr>
                            <td style="width: 40%; padding: 10px;">
                                <div class="hospital">
                                    <div style="font-size: 12px; color: #6c757d;">FROM</div>
                                    <div style="font-size: 16px; font-weight: bold;">{source_name}</div>
                                </div>
                            </td>
                            <td style="width: 20%; font-size: 30px; color: {urgency_color};">→</td>
                            <td style="width: 40%; padding: 10px;">
                                <div class="hospital">
                                    <div style="font-size: 12px; color: #6c757d;">TO</div>
                                    <div style="font-size: 16px; font-weight: bold;">{target_name}</div>
                                </div>
                            </td>
                        </tr>
                    </table>
                </div>
                
                <h3>📋 Transfer Details</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Transfer ID</div>
                        <div class="info-value" style="font-size: 14px;">{transfer_id}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Transfer Reason</div>
                        <div class="info-value" style="font-size: 14px;">{transfer_reason}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Patient Age Bucket</div>
                        <div class="info-value">{patient_age_bucket}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Risk Level</div>
                        <div class="info-value"><span class="risk-badge">{risk_level}</span></div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Est. Transport Time</div>
                        <div class="info-value">{estimated_transport_minutes} minutes</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Timestamp</div>
                        <div class="info-value" style="font-size: 14px;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</div>
                    </div>
                </div>
                
                <div class="bed-info">
                    <h4 style="margin-top: 0;">🛏️ Bed Reservation</h4>
                    <p><strong>Bed ID:</strong> {bed_id}</p>
                    <p><strong>Unit:</strong> {bed_unit}</p>
                    <p><strong>Reserved Until:</strong> {reserved_until}</p>
                </div>
                
                <div class="privacy-note">
                    <strong>🔒 Privacy Compliance</strong><br>
                    This transfer uses HIPAA-compliant privacy-preserving protocols:
                    <ul style="margin: 10px 0;">
                        <li>Patient data anonymized (HIPAA Safe Harbor)</li>
                        <li>K-anonymity (k ≥ 5)</li>
                        <li>Differential privacy (ε = 0.1)</li>
                        <li>Full audit trail maintained</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p><strong>HIMAS - Healthcare Intelligence Multi-Agent System</strong></p>
                <p>Federated Learning • Privacy-Preserving • HIPAA Compliant</p>
                <p style="color: #adb5bd;">This is an automated notification. Please do not reply.</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html


def send_test_notification(recipient_email: str) -> Dict[str, Any]:
    """
    Send a test notification to verify email configuration.

    Args:
        recipient_email: Email address to send test to

    Returns:
        dict with test result
    """
    return send_transfer_notification(
        transfer_id="TEST_" + datetime.now().strftime('%Y%m%d_%H%M%S'),
        source_hospital="hospital_a",
        target_hospital="hospital_b",
        transfer_reason="TEST - Email Configuration Verification",
        urgency="LOW",
        patient_age_bucket="N/A (Test)",
        risk_level="LOW",
        bed_reservation={
            "bed_id": "TEST_BED",
            "unit": "TEST_UNIT",
            "reserved_until": "N/A"
        },
        estimated_transport_minutes=0,
        additional_recipients=[recipient_email]
    )
