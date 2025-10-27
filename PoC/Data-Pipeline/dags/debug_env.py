"""
Debug DAG to verify environment variables are loaded correctly.
"""
from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
import os
import logging

logger = logging.getLogger(__name__)


def check_environment_variables(**context):
    """Check if all required environment variables are set."""
    logger.info("=" * 60)
    logger.info("ENVIRONMENT VARIABLES CHECK")
    logger.info("=" * 60)

    # List of variables to check
    env_vars = {
        'SMTP Settings': [
            'AIRFLOW__SMTP__SMTP_HOST',
            'AIRFLOW__SMTP__SMTP_PORT',
            'AIRFLOW__SMTP__SMTP_USER',
            'AIRFLOW__SMTP__SMTP_PASSWORD',
            'AIRFLOW__SMTP__SMTP_STARTTLS',
        ],
        'Email Settings': [
            'AIRFLOW__EMAIL__SUBJECT_TEMPLATE',
            'AIRFLOW__EMAIL__HTML_CONTENT_TEMPLATE',
        ],
        'HIMAS Custom': [
            'GOOGLE_CLOUD_PROJECT',
            'USE_GCS',
            'GCS_REPORTS_BUCKET',
            'ALERT_EMAILS',
        ]
    }

    all_set = True

    for category, vars_list in env_vars.items():
        logger.info(f"\n{category}:")
        logger.info("-" * 40)
        for var in vars_list:
            value = os.getenv(var)
            if value:
                # Hide password
                if 'PASSWORD' in var:
                    logger.info(f"  ✅ {var}: ***SET***")
                else:
                    logger.info(f"  ✅ {var}: {value}")
            else:
                logger.error(f"  ❌ {var}: NOT SET")
                all_set = False

    logger.info("=" * 60)

    if all_set:
        logger.info("✅ All environment variables are set correctly!")
        return "SUCCESS"
    else:
        logger.error("❌ Some environment variables are missing!")
        raise ValueError("Missing environment variables - check logs above")


def test_config_import(**context):
    """Test if config.py can import environment variables."""
    logger.info("=" * 60)
    logger.info("TESTING CONFIG.PY IMPORTS")
    logger.info("=" * 60)

    try:
        from utils.config import PipelineConfig

        logger.info("✅ PipelineConfig imported successfully")
        logger.info(f"  PROJECT_ID: {PipelineConfig.PROJECT_ID}")
        logger.info(f"  LOCATION: {PipelineConfig.LOCATION}")
        logger.info(f"  USE_GCS: {PipelineConfig.USE_GCS}")
        logger.info(f"  GCS_BUCKET: {PipelineConfig.GCS_BUCKET}")
        logger.info(f"  ALERT_EMAIL: {PipelineConfig.ALERT_EMAILS}")

        logger.info("=" * 60)
        return "CONFIG LOADED"
    except Exception as e:
        logger.error(f"❌ Failed to import config: {str(e)}")
        logger.exception("Full error:")
        raise


with DAG(
    dag_id='debug_environment',
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=['debug', 'test'],
    doc_md="""
    # Debug Environment Variables
    
    This DAG checks if all environment variables are properly loaded.
    Run this first to verify your configuration.
    """
) as dag:

    check_env = PythonOperator(
        task_id='check_environment_variables',
        python_callable=check_environment_variables,
    )

    test_config = PythonOperator(
        task_id='test_config_import',
        python_callable=test_config_import,
    )

    check_env >> test_config
