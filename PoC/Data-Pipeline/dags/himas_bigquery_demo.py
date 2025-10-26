"""
HIMAS BigQuery Data Pipeline - Modularized Demo Edition with DVC

Features:
- Modular configuration (utils/config.py)
- Hybrid storage support (utils/storage.py)
- SQL file utilities (utils/sql_utils.py)
- Data validation (utils/validation.py)
- Email callbacks (utils/email_callbacks.py)
- DVC data versioning (utils/dvc_handler.py)
- Fast import (<5 seconds)
- Works locally or with GCS
"""
from utils.email_callbacks import send_success_email
from utils.validation import DataValidator
from utils.sql_utils import SQLFileLoader
from utils.storage import StorageHandler
from utils.config import PipelineConfig
from utils.dvc_handler import DVCHandler
from datetime import datetime
from airflow.sdk import DAG, task_group
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCreateEmptyDatasetOperator,
    BigQueryInsertJobOperator,
)


# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================

# Configuration
config = PipelineConfig()

# Storage handler
storage = StorageHandler(
    use_gcs=config.USE_GCS,
    gcs_bucket=config.GCS_BUCKET,
    local_dir=config.REPORTS_DIR,
    project_id=config.PROJECT_ID
)

# SQL loader
sql_loader = SQLFileLoader(sql_dir=config.SQL_DIR)

# Data validator
validator = DataValidator(
    project_id=config.PROJECT_ID,
    location=config.LOCATION
)

# DVC handler
dvc_handler = DVCHandler(
    repo_path="/opt/airflow",
    use_gcs=config.USE_GCS,
    gcs_bucket=config.GCS_BUCKET,
    project_id=config.PROJECT_ID
)

# ============================================================================
# TASK FUNCTIONS (Parameterized)
# ============================================================================


def verify_data_integrity_task(**context):
    """
    Task function: Verify data integrity using modular validator.

    Uses:
        - validator (DataValidator)
        - storage (StorageHandler)
    """
    return validator.verify_data_integrity(storage, **context)


def generate_statistics_task(**context):
    """
    Task function: Generate statistics using modular validator.

    Uses:
        - validator (DataValidator)
        - storage (StorageHandler)
    """
    return validator.generate_statistics(storage, **context)


def version_reports_task(**context):
    """
    Task function: Version the reports directory with DVC.

    Uses:
        - dvc_handler (DVCHandler)
    """
    import logging
    logger = logging.getLogger(__name__)

    # Get run ID
    run_id = context['run_id']

    logger.info(f"Versioning reports for run_id: {run_id}")

    # Create version metadata
    metadata = dvc_handler.create_version_metadata(
        run_id=run_id,
        context=context
    )
    logger.info(f"Created version metadata: {metadata}")

    # Version reports
    logger.info("Versioning reports directory...")
    success = dvc_handler.version_reports()

    if success:
        status = dvc_handler.get_data_status()
        logger.info(f"DVC Status: {status}")
        return {
            "success": True,
            "metadata": metadata,
            "status": status
        }
    else:
        raise Exception("Failed to version reports with DVC")


def version_all_data_task(**context):
    """
    Task function: Version all data directories with DVC.

    Uses:
        - dvc_handler (DVCHandler)
    """
    import logging
    logger = logging.getLogger(__name__)

    # Get run ID
    run_id = context['run_id']

    # Version all data
    logger.info("Versioning all data directories...")
    success = dvc_handler.version_all_data()

    if success:
        status = dvc_handler.get_data_status()
        logger.info(f"DVC Status after versioning: {status}")

        # Push XCom for downstream tasks
        context['task_instance'].xcom_push(key='dvc_version', value=run_id)

        return {
            "success": True,
            "run_id": run_id,
            "status": status,
            "remote_type": "gcs" if dvc_handler.use_gcs else "local"
        }
    else:
        raise Exception("Failed to version data with DVC")


def version_bigquery_layers_task(**context):
    """
    Task function: Export and version BigQuery layers.

    Uses:
        - dvc_handler (DVCHandler)
        - config (PipelineConfig)
    """
    import logging
    logger = logging.getLogger(__name__)

    run_id = context['run_id']

    logger.info(
        f"Exporting and versioning BigQuery layers for run_id: {run_id}")

    # Export and version BigQuery tables
    success = dvc_handler.version_bigquery_layers(
        project_id=config.PROJECT_ID
    )

    if success:
        logger.info("Successfully versioned BigQuery layers")
        return {
            "success": True,
            "run_id": run_id,
            "layers": ["curated", "federated", "verification"]
        }
    else:
        raise Exception("Failed to version BigQuery layers")


# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(
    dag_id='himas_bigquery_demo',
    default_args={
        'owner': 'himas',
        'depends_on_past': False,
        'email': config.ALERT_EMAILS,
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 0,
    },
    description='HIMAS BigQuery Pipeline with DVC Versioning',
    schedule=None,  # Manual trigger for demos
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['himas', 'bigquery', 'demo', 'modular', 'dvc'],
    on_success_callback=send_success_email,
    doc_md=f"""
    # HIMAS Data Pipeline - Demo Edition with DVC

    **Storage**: {storage.get_storage_info()}
    **Project**: {config.PROJECT_ID}
    **DVC Remote**: {'GCS' if config.USE_GCS else 'Local'}

    This DAG processes MIMIC-IV demo data for federated learning with data versioning.

    ## Features
    - Modular architecture
    - Hybrid storage (local/GCS)
    - Native BigQuery operators
    - Atomic SQL tasks
    - Email alerts
    - **DVC data versioning**

    ## DVC Integration
    - Automatic versioning of reports and datasets
    - Support for local and GCS remotes
    - Version metadata tracking
    - Data integrity validation

    ## Outputs
    - Curated dimensional tables
    - Federated hospital views
    - Data quality reports
    - **Versioned datasets (.dvc files)**
    """
) as dag:

    # ========================================================================
    # TASK GROUP 1: CREATE DATASETS
    # ========================================================================

    @task_group(tooltip='Create BigQuery datasets for all layers')
    def create_datasets():
        """Create BigQuery datasets for all pipeline layers."""
        for dataset_config in config.DATASETS:
            BigQueryCreateEmptyDatasetOperator(
                task_id=f"create_{dataset_config['dataset_id']}",
                dataset_id=dataset_config['dataset_id'],
                project_id=config.PROJECT_ID,
                location=dataset_config['location'],
                gcp_conn_id=config.GCP_CONN_ID,
                if_exists='skip',
                exists_ok=True
            )

    # ========================================================================
    # TASK GROUP 2: CURATED LAYER
    # ========================================================================

    @task_group(tooltip='Create dimensional model tables')
    def curated_layer():
        """
        Create dimensional model tables.

        Creates:
        - patient_split_assignment
        - dim_patient
        - fact_hospital_admission
        - fact_icu_stay
        - fact_transfers
        - clinical_features
        """
        curated_sql_files = sql_loader.get_layer_files('curated_layer')
        curated_tasks = []

        for sql_file in curated_sql_files:
            file_stem = sql_file.stem

            task = BigQueryInsertJobOperator(
                task_id=f"create_{file_stem}",
                configuration={
                    "query": {
                        "query": sql_file.read_text(),
                        "useLegacySql": False,
                    }
                },
                location=config.LOCATION,
                gcp_conn_id=config.GCP_CONN_ID,
            )

            curated_tasks.append(task)

        # Chain sequentially (order matters for dimensional model)
        if len(curated_tasks) > 1:
            for i in range(len(curated_tasks) - 1):
                curated_tasks[i] >> curated_tasks[i + 1]

    # ========================================================================
    # TASK GROUP 3: FEDERATED LAYER
    # ========================================================================

    @task_group(tooltip='Create hospital-specific federated views')
    def federated_layer():
        """
        Create federated hospital data.

        Creates:
        - hospital_a_data (40% of patients)
        - hospital_b_data (35% of patients)
        - hospital_c_data (25% of patients)
        """
        federated_sql_files = sql_loader.get_layer_files('federated_layer')

        for sql_file in federated_sql_files:
            file_stem = sql_file.stem

            BigQueryInsertJobOperator(
                task_id=f"create_{file_stem}",
                configuration={
                    "query": {
                        "query": sql_file.read_text(),
                        "useLegacySql": False,
                    }
                },
                location=config.LOCATION,
                gcp_conn_id=config.GCP_CONN_ID,
            )

    # ========================================================================
    # TASK GROUP 4: VERIFICATION LAYER
    # ========================================================================

    @task_group(tooltip='Create data quality check views')
    def verification_layer():
        """
        Create verification views.

        Creates:
        - data_leakage_check
        - dataset_statistics
        """
        verification_sql_files = sql_loader.get_layer_files(
            'verification_layer')

        for sql_file in verification_sql_files:
            file_stem = sql_file.stem

            BigQueryInsertJobOperator(
                task_id=f"create_{file_stem}",
                configuration={
                    "query": {
                        "query": sql_file.read_text(),
                        "useLegacySql": False,
                    }
                },
                location=config.LOCATION,
                gcp_conn_id=config.GCP_CONN_ID,
            )

    # ========================================================================
    # TASK GROUP 5: QUALITY CHECKS & REPORTING
    # ========================================================================

    @task_group(tooltip='Verify data integrity and generate reports')
    def quality_checks():
        """Run data quality checks and generate reports."""
        verify_integrity = PythonOperator(
            task_id='verify_data_integrity',
            python_callable=verify_data_integrity_task,
        )

        generate_stats = PythonOperator(
            task_id='generate_statistics',
            python_callable=generate_statistics_task,
        )

        verify_integrity >> generate_stats

    # ========================================================================
    # TASK GROUP 6: DVC VERSIONING
    # ========================================================================

    @task_group(tooltip='Version datasets and reports with DVC')
    def version_with_dvc():
        """Version all generated data with DVC."""

        version_bigquery = PythonOperator(
            task_id='version_bigquery_layers',
            python_callable=version_bigquery_layers_task,
        )

        version_reports = PythonOperator(
            task_id='version_reports',
            python_callable=version_reports_task,
        )

        version_all_data = PythonOperator(
            task_id='version_all_data',
            python_callable=version_all_data_task,
        )

        version_bigquery >> version_reports >> version_all_data

    # ========================================================================
    # PIPELINE DEPENDENCIES
    # ========================================================================

    (create_datasets() >> curated_layer() >>
     federated_layer() >> verification_layer() >> quality_checks() >> version_with_dvc())
