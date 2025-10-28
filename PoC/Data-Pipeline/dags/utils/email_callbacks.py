"""
Email Callbacks for HIMAS Pipeline
"""

from airflow.utils.email import send_email
from utils.config import PipelineConfig
import logging

logger = logging.getLogger(__name__)

# Configuration
config = PipelineConfig()

# Email list for alerts
ALERT_EMAILS = config.ALERT_EMAILS

def send_success_email(context):
    """
    Send success email when DAG completes successfully.
    Uses Airflow's native send_email function.
    """
    try:
        # Get DAG information
        dag = context.get('dag')
        dag_id = dag.dag_id if dag else 'unknown'

        # Get dates
        logical_date = context.get('logical_date')
        ds = context.get('ds', 'N/A')
        ts = context.get('ts', 'N/A')
        run_id = context.get('run_id', 'N/A')

        # Format logical date for display
        logical_date_str = logical_date.strftime(
            '%Y-%m-%d %H:%M:%S UTC') if logical_date else ds

        # Get task count from DAG
        total_tasks = len(dag.task_ids) if dag else 0
        task_list = ', '.join(dag.task_ids) if dag else 'N/A'

        # Email subject
        subject = f"✅ HIMAS Pipeline Success: {dag_id}"

        # Email HTML content
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
                .header {{ 
                    background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                    color: white; padding: 30px; text-align: center;
                }}
                .content {{ padding: 30px; background-color: #f8f9fa; }}
                .success-banner {{
                    background-color: #d4edda;
                    border: 2px solid #28a745;
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 5px solid #28a745;
                    margin: 15px 0;
                }}
                .stat-label {{ 
                    color: #6c757d; 
                    font-size: 14px; 
                    font-weight: bold;
                    text-transform: uppercase;
                }}
                .stat-value {{ 
                    font-size: 18px; 
                    color: #28a745; 
                    margin: 5px 0;
                }}
                .info-box {{ 
                    background-color: #e7f3ff; 
                    border-left: 4px solid #0066cc; 
                    padding: 15px; 
                    margin: 15px 0;
                    border-radius: 5px;
                }}
                .task-list {{
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    font-family: monospace;
                    font-size: 12px;
                }}
                .btn {{
                    display: inline-block;
                    background-color: #28a745;
                    color: white;
                    padding: 12px 24px;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: bold;
                    margin: 10px 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎉 Pipeline Completed Successfully!</h1>
                <p>HIMAS Federated Learning Data Pipeline</p>
            </div>

            <div class="content">
                <div class="success-banner">
                    <h2 style="color: #28a745; margin: 0;">✓ All Tasks Successful</h2>
                    <p>Federated learning data is ready for deployment</p>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Total Tasks Completed</div>
                    <div class="stat-value">{total_tasks} tasks</div>
                </div>

                <div class="info-box">
                    <h3>📋 Pipeline Details</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px; font-weight: bold; width: 30%;">DAG ID:</td>
                            <td style="padding: 8px;">{dag_id}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">Run ID:</td>
                            <td style="padding: 8px; font-size: 12px;">{run_id}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">Logical Date:</td>
                            <td style="padding: 8px;">{logical_date_str}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">Timestamp:</td>
                            <td style="padding: 8px;">{ts}</td>
                        </tr>
                    </table>
                </div>

                <div class="info-box">
                    <h3>✅ Generated Outputs</h3>
                    <ul style="line-height: 1.8;">
                        <li><strong>Curated Layer:</strong> Dimensional tables in BigQuery</li>
                        <li><strong>Federated Layer:</strong> Hospital-specific views (3 hospitals)</li>
                        <li><strong>Verification:</strong> Data integrity verified (zero leakage)</li>
                        <li><strong>Reports:</strong> Statistics and quality reports generated</li>
                    </ul>
                </div>

                <div class="info-box">
                    <h3>🔧 Tasks Executed</h3>
                    <div class="task-list">{task_list}</div>
                </div>

                <div style="text-align: center; margin: 30px 0;">
                    <a href="http://localhost:8080/dags/{dag_id}/grid" class="btn">
                        📊 View Pipeline in Airflow
                    </a>
                </div>
            </div>

            <div style="background-color: #343a40; color: white; padding: 20px; text-align: center;">
                <p style="margin: 5px 0;"><strong>HIMAS - Healthcare Intelligence Multi-Agent System</strong></p>
                <p style="margin: 5px 0; font-size: 12px;">Automated Pipeline Notification</p>
                <p style="margin: 5px 0; font-size: 11px; color: #adb5bd;">
                    🔒 Federated Learning • 🏥 MIMIC-IV Dataset • ☁️ Google Cloud Platform
                </p>
            </div>
        </body>
        </html>
        """

        # Send email using Airflow's native function
        send_email(
            to=ALERT_EMAILS,
            subject=subject,
            html_content=html_content
        )

        logger.info(f"✅ Success email sent for DAG: {dag_id}")

    except Exception as e:
        # Log error but don't fail the DAG
        logger.error(f"❌ Failed to send success email: {str(e)}")
        logger.exception("Full traceback:")
