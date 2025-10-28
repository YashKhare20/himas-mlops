"""
HIMAS BigQuery Data Pipeline - Schema & Statistics Validation

Automated schema extraction, statistics generation, and data quality validation.

Features:
- Automated schema extraction from BigQuery
- Field-level statistics computation
- Schema drift detection with baselines
- Data quality validation with thresholds
- DVC data versioning
- Email alerting
"""
from datetime import datetime
from airflow.sdk import DAG, task_group
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCreateEmptyDatasetOperator,
    BigQueryInsertJobOperator,
)

from utils.config import PipelineConfig
from utils.email_callbacks import send_success_email
from utils.task_functions import (
    create_extract_schemas_task_function,
    create_compute_statistics_task_function,
    create_detect_drift_task_function,
    create_validate_quality_task_function,
    create_quality_summary_task_function,
    create_dvc_version_reports_task_function,
    create_dvc_version_all_data_task_function,
    create_dvc_version_bigquery_task_function
)


# ============================================================================
# LAZY INITIALIZATION
# ============================================================================

def get_schema_validator():
    """Initialize SchemaValidator for schema extraction and validation."""
    from utils.schema_validator import SchemaValidator
    config = PipelineConfig()
    return SchemaValidator(
        project_id=config.PROJECT_ID,
        location=config.LOCATION
    )


def get_storage_handler():
    """Initialize StorageHandler for local/GCS storage."""
    from utils.storage import StorageHandler
    config = PipelineConfig()
    return StorageHandler(
        use_gcs=config.USE_GCS,
        gcs_bucket=config.GCS_BUCKET,
        local_dir=config.DATA_DIR,
        project_id=config.PROJECT_ID
    )


def get_dvc_handler():
    """Initialize DVCHandler for data versioning."""
    from utils.dvc_handler import DVCHandler
    config = PipelineConfig()
    return DVCHandler(
        repo_path="/opt/airflow",
        use_gcs=config.USE_GCS,
        gcs_bucket=config.GCS_BUCKET,
        project_id=config.PROJECT_ID
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

config = PipelineConfig()


# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(
    dag_id='himas_bigquery_schema_validation',
    default_args={
        'owner': 'himas',
        'depends_on_past': False,
        'email': config.ALERT_EMAILS,
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 0,
    },
    description='HIMAS Pipeline with Schema & Statistics Validation',
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['himas', 'bigquery', 'schema-validation', 'dvc', 'mlops'],
    on_success_callback=send_success_email,
    doc_md=f"""
    # HIMAS Data Pipeline - Schema & Statistics Validation

    **Project**: {config.PROJECT_ID}  
    **Location**: {config.LOCATION}  
    **DVC Remote**: {'GCS' if config.USE_GCS else 'Local'}

    ## Pipeline Flow
    
    1. **create_datasets()** - Create BigQuery datasets
    2. **curated_layer()** - Build dimensional model (6 tables)
    3. **federated_layer()** - Create hospital partitions (3 hospitals)
    4. **verification_layer()** - Quality check views (2 tables)
    5. **schema_and_statistics()** - Schema validation
       - Extract schemas from all tables
       - Compute field-level statistics
       - Detect schema drift vs baseline
       - Validate data quality
       - Generate summary report
    6. **version_with_dvc()** - Version all outputs
    
    ## Outputs
    
    All outputs stored in `/data/` directory (local or GCS based on USE_GCS flag):
    
    - `/data/schemas/` - Table schemas (JSON)
    - `/data/statistics/` - Field statistics (JSON)
    - `/data/drift/` - Drift reports (JSON)
    - `/data/validation/` - Quality validation (JSON)
    - `/data/reports/` - Summary reports (JSON)
    
    ## Validation Thresholds
    
    - Max row count change: 50%
    - Max null rate per field: 30%
    - Min distinct ratio: 1%
    
    See README.md for complete documentation.
    """
) as dag:

    # ========================================================================
    # TASK GROUP 1: CREATE DATASETS
    # ========================================================================

    @task_group(tooltip='Create BigQuery datasets')
    def create_datasets():
        """Create BigQuery datasets for all layers."""
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

    @task_group(tooltip='Create dimensional model')
    def curated_layer():
        """Create curated layer tables (sequential execution)."""
        curated_tasks = []

        for table_name in config.CURATED_TABLES:
            sql_file = config.SQL_DIR / 'curated_layer' / f'{table_name}.sql'

            task = BigQueryInsertJobOperator(
                task_id=f"create_{table_name}",
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

        # Chain sequentially
        if len(curated_tasks) > 1:
            for i in range(len(curated_tasks) - 1):
                curated_tasks[i] >> curated_tasks[i + 1]

    # ========================================================================
    # TASK GROUP 3: FEDERATED LAYER
    # ========================================================================

    @task_group(tooltip='Create hospital partitions')
    def federated_layer():
        """Create federated hospital data (parallel execution)."""
        for table_name in config.FEDERATED_TABLES:
            sql_file = config.SQL_DIR / 'federated_layer' / f'{table_name}.sql'

            BigQueryInsertJobOperator(
                task_id=f"create_{table_name}",
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

    @task_group(tooltip='Create quality check views')
    def verification_layer():
        """Create verification views (parallel execution)."""
        for table_name in config.VERIFICATION_TABLES:
            sql_file = config.SQL_DIR / \
                'verification_layer' / f'{table_name}.sql'

            BigQueryInsertJobOperator(
                task_id=f"create_{table_name}",
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
    # TASK GROUP 5: SCHEMA & STATISTICS VALIDATION
    # ========================================================================

    @task_group(tooltip='Schema and statistics validation')
    def schema_and_statistics():
        """
        Validate schemas and statistics.

        Tasks:
        1. Extract schemas from BigQuery tables
        2. Compute field-level statistics
        3. Detect schema drift vs baseline
        4. Validate data quality against thresholds
        5. Generate comprehensive summary
        """
        extract_schemas = PythonOperator(
            task_id='extract_all_schemas',
            python_callable=create_extract_schemas_task_function(
                schema_validator=get_schema_validator(),
                config=config,
                storage_handler=get_storage_handler()
            ),
        )

        compute_stats = PythonOperator(
            task_id='compute_all_statistics',
            python_callable=create_compute_statistics_task_function(
                schema_validator=get_schema_validator(),
                config=config,
                storage_handler=get_storage_handler()
            ),
        )

        detect_drift = PythonOperator(
            task_id='detect_schema_drift',
            python_callable=create_detect_drift_task_function(
                schema_validator=get_schema_validator(),
                config=config,
                storage_handler=get_storage_handler()
            ),
        )

        validate_quality = PythonOperator(
            task_id='validate_data_quality',
            python_callable=create_validate_quality_task_function(
                schema_validator=get_schema_validator(),
                config=config,
                storage_handler=get_storage_handler()
            ),
        )

        generate_summary = PythonOperator(
            task_id='generate_quality_summary',
            python_callable=create_quality_summary_task_function(
                schema_validator=get_schema_validator(),
                config=config,
                storage_handler=get_storage_handler()
            ),
        )

        # Dependencies: extract/compute → drift → validate → summary
        [extract_schemas, compute_stats] >> detect_drift >> validate_quality >> generate_summary

    # ========================================================================
    # TASK GROUP 6: DVC VERSIONING
    # ========================================================================

    @task_group(tooltip='Version data with DVC')
    def version_with_dvc():
        """Version all outputs with DVC."""
        version_bigquery = PythonOperator(
            task_id='version_bigquery_layers',
            python_callable=create_dvc_version_bigquery_task_function(
                dvc_handler=get_dvc_handler(),
                config=config
            ),
        )

        version_reports = PythonOperator(
            task_id='version_reports',
            python_callable=create_dvc_version_reports_task_function(
                dvc_handler=get_dvc_handler()
            ),
        )

        version_all_data = PythonOperator(
            task_id='version_all_data',
            python_callable=create_dvc_version_all_data_task_function(
                dvc_handler=get_dvc_handler()
            ),
        )

        # Sequential versioning
        version_bigquery >> version_reports >> version_all_data

    # ========================================================================
    # PIPELINE DEPENDENCIES
    # ========================================================================

    (
        create_datasets()
        >> curated_layer()
        >> federated_layer()
        >> verification_layer()
        >> schema_and_statistics()
        >> version_with_dvc()
    )
