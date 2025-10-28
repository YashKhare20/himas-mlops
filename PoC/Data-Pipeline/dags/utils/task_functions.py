"""
Task Functions for HIMAS BigQuery Pipeline

Includes DVC versioning tasks and Schema & Statistics validation tasks.
"""
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.schema_validator import SchemaValidator
    from utils.storage import StorageHandler
    from utils.dvc_handler import DVCHandler
    from utils.config import PipelineConfig

logger = logging.getLogger(__name__)


# ============================================================================
# SCHEMA & STATISTICS VALIDATION TASKS
# ============================================================================


def extract_all_schemas_task(schema_validator, config, storage_handler, **context) -> Dict[str, Any]:
    """
    Extract schemas from all BigQuery layers using config.

    Args:
        schema_validator: SchemaValidator instance
        config: PipelineConfig instance
        storage_handler: StorageHandler instance
        **context: Airflow context

    Returns:
        Dictionary with extraction results
    """
    from utils.schema_utils import extract_all_layer_schemas

    run_id = context['run_id']
    return extract_all_layer_schemas(schema_validator, run_id, storage_handler)


def compute_all_statistics_task(schema_validator, config, storage_handler, **context) -> Dict[str, Any]:
    """
    Compute statistics for all BigQuery layers using config.

    Args:
        schema_validator: SchemaValidator instance
        config: PipelineConfig instance
        storage_handler: StorageHandler instance
        **context: Airflow context

    Returns:
        Dictionary with computation results
    """
    from utils.schema_utils import compute_all_layer_statistics

    run_id = context['run_id']
    return compute_all_layer_statistics(schema_validator, run_id, storage_handler)


def detect_schema_drift_task(schema_validator, config, storage_handler, **context) -> Dict[str, Any]:
    """
    Detect schema drift by comparing current schemas with baseline.

    Args:
        schema_validator: SchemaValidator instance
        config: PipelineConfig instance
        storage_handler: StorageHandler instance
        **context: Airflow context

    Returns:
        Dictionary with drift detection results
    """
    from utils.schema_utils import detect_schema_drift_all_layers

    run_id = context['run_id']
    return detect_schema_drift_all_layers(schema_validator, run_id, storage_handler)


def validate_data_quality_task(schema_validator, config, storage_handler, **context) -> Dict[str, Any]:
    """
    Validate data quality against baseline and thresholds.

    Args:
        schema_validator: SchemaValidator instance
        config: PipelineConfig instance
        storage_handler: StorageHandler instance
        **context: Airflow context

    Returns:
        Dictionary with validation results
    """
    from utils.schema_utils import validate_data_quality_all_layers

    run_id = context['run_id']

    # Custom thresholds for HIMAS healthcare data
    custom_thresholds = {
        "row_count_change_pct": 50.0,    # Max 50% change in row count
        "null_rate_threshold": 0.3,       # Max 30% nulls per field
        "distinct_ratio_min": 0.01        # Min 1% distinct values
    }

    return validate_data_quality_all_layers(
        schema_validator,
        run_id,
        storage_handler,
        custom_thresholds=custom_thresholds
    )


def generate_quality_summary_task(schema_validator, config, storage_handler, **context) -> Dict[str, Any]:
    """
    Generate comprehensive quality summary report.

    Args:
        schema_validator: SchemaValidator instance
        config: PipelineConfig instance
        storage_handler: StorageHandler instance
        **context: Airflow context

    Returns:
        Dictionary with summary results
    """
    from utils.schema_utils import generate_comprehensive_quality_summary

    run_id = context['run_id']
    return generate_comprehensive_quality_summary(run_id, config, storage_handler)


# ============================================================================
# DVC VERSIONING TASKS
# ============================================================================


def version_reports_task(dvc_handler, **context) -> Dict[str, Any]:
    """
    Version the reports directory with DVC.

    Args:
        dvc_handler: DVCHandler instance
        **context: Airflow context

    Returns:
        Dictionary with versioning results
    """
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


def version_all_data_task(dvc_handler, **context) -> Dict[str, Any]:
    """
    Version all data directories with DVC.

    Args:
        dvc_handler: DVCHandler instance
        **context: Airflow context

    Returns:
        Dictionary with versioning results
    """
    run_id = context['run_id']

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


def version_bigquery_layers_task(dvc_handler, config, **context) -> Dict[str, Any]:
    """
    Export and version BigQuery layers.

    Args:
        dvc_handler: DVCHandler instance
        config: PipelineConfig instance
        **context: Airflow context

    Returns:
        Dictionary with versioning results
    """
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
# WRAPPER FUNCTIONS FOR AIRFLOW TASKS (WITH LAZY INITIALIZATION)
# ============================================================================


def create_extract_schemas_task_function(
    schema_validator: Optional['SchemaValidator'],
    config: 'PipelineConfig',
    storage_handler: Optional['StorageHandler']
):
    """
    Factory function: Create schema extraction task function.

    Supports lazy initialization - if schema_validator or storage_handler are None,
    they will be initialized at task runtime (not DAG parse time).
    This allows DAG to be parsed in CI/CD without GCP credentials.

    Args:
        schema_validator: SchemaValidator instance or None for lazy init
        config: PipelineConfig instance
        storage_handler: StorageHandler instance or None for lazy init

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        # Lazy initialization at runtime (not DAG parse time)
        if schema_validator is None:
            from utils.schema_validator import SchemaValidator
            validator = SchemaValidator(
                project_id=config.PROJECT_ID,
                location=config.LOCATION
            )
        else:
            validator = schema_validator

        if storage_handler is None:
            from utils.storage import StorageHandler
            handler = StorageHandler(
                use_gcs=config.USE_GCS,
                gcs_bucket=config.GCS_BUCKET,
                local_dir=config.DATA_DIR,
                project_id=config.PROJECT_ID
            )
        else:
            handler = storage_handler

        return extract_all_schemas_task(
            schema_validator=validator,
            config=config,
            storage_handler=handler,
            **context
        )
    return task_function


def create_compute_statistics_task_function(
    schema_validator: Optional['SchemaValidator'],
    config: 'PipelineConfig',
    storage_handler: Optional['StorageHandler']
):
    """
    Factory function: Create statistics computation task function.

    Supports lazy initialization for CI/CD compatibility.

    Args:
        schema_validator: SchemaValidator instance or None for lazy init
        config: PipelineConfig instance
        storage_handler: StorageHandler instance or None for lazy init

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        # Lazy initialization at runtime
        if schema_validator is None:
            from utils.schema_validator import SchemaValidator
            validator = SchemaValidator(
                project_id=config.PROJECT_ID,
                location=config.LOCATION
            )
        else:
            validator = schema_validator

        if storage_handler is None:
            from utils.storage import StorageHandler
            handler = StorageHandler(
                use_gcs=config.USE_GCS,
                gcs_bucket=config.GCS_BUCKET,
                local_dir=config.DATA_DIR,
                project_id=config.PROJECT_ID
            )
        else:
            handler = storage_handler

        return compute_all_statistics_task(
            schema_validator=validator,
            config=config,
            storage_handler=handler,
            **context
        )
    return task_function


def create_detect_drift_task_function(
    schema_validator: Optional['SchemaValidator'],
    config: 'PipelineConfig',
    storage_handler: Optional['StorageHandler']
):
    """
    Factory function: Create drift detection task function.

    Supports lazy initialization for CI/CD compatibility.

    Args:
        schema_validator: SchemaValidator instance or None for lazy init
        config: PipelineConfig instance
        storage_handler: StorageHandler instance or None for lazy init

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        # Lazy initialization at runtime
        if schema_validator is None:
            from utils.schema_validator import SchemaValidator
            validator = SchemaValidator(
                project_id=config.PROJECT_ID,
                location=config.LOCATION
            )
        else:
            validator = schema_validator

        if storage_handler is None:
            from utils.storage import StorageHandler
            handler = StorageHandler(
                use_gcs=config.USE_GCS,
                gcs_bucket=config.GCS_BUCKET,
                local_dir=config.DATA_DIR,
                project_id=config.PROJECT_ID
            )
        else:
            handler = storage_handler

        return detect_schema_drift_task(
            schema_validator=validator,
            config=config,
            storage_handler=handler,
            **context
        )
    return task_function


def create_validate_quality_task_function(
    schema_validator: Optional['SchemaValidator'],
    config: 'PipelineConfig',
    storage_handler: Optional['StorageHandler']
):
    """
    Factory function: Create quality validation task function.

    Supports lazy initialization for CI/CD compatibility.

    Args:
        schema_validator: SchemaValidator instance or None for lazy init
        config: PipelineConfig instance
        storage_handler: StorageHandler instance or None for lazy init

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        # Lazy initialization at runtime
        if schema_validator is None:
            from utils.schema_validator import SchemaValidator
            validator = SchemaValidator(
                project_id=config.PROJECT_ID,
                location=config.LOCATION
            )
        else:
            validator = schema_validator

        if storage_handler is None:
            from utils.storage import StorageHandler
            handler = StorageHandler(
                use_gcs=config.USE_GCS,
                gcs_bucket=config.GCS_BUCKET,
                local_dir=config.DATA_DIR,
                project_id=config.PROJECT_ID
            )
        else:
            handler = storage_handler

        return validate_data_quality_task(
            schema_validator=validator,
            config=config,
            storage_handler=handler,
            **context
        )
    return task_function


def create_quality_summary_task_function(
    schema_validator: Optional['SchemaValidator'],
    config: 'PipelineConfig',
    storage_handler: Optional['StorageHandler']
):
    """
    Factory function: Create quality summary task function.

    Supports lazy initialization for CI/CD compatibility.

    Args:
        schema_validator: SchemaValidator instance or None for lazy init
        config: PipelineConfig instance
        storage_handler: StorageHandler instance or None for lazy init

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        # Lazy initialization at runtime
        if schema_validator is None:
            from utils.schema_validator import SchemaValidator
            validator = SchemaValidator(
                project_id=config.PROJECT_ID,
                location=config.LOCATION
            )
        else:
            validator = schema_validator

        if storage_handler is None:
            from utils.storage import StorageHandler
            handler = StorageHandler(
                use_gcs=config.USE_GCS,
                gcs_bucket=config.GCS_BUCKET,
                local_dir=config.DATA_DIR,
                project_id=config.PROJECT_ID
            )
        else:
            handler = storage_handler

        return generate_quality_summary_task(
            schema_validator=validator,
            config=config,
            storage_handler=handler,
            **context
        )
    return task_function


def create_dvc_version_reports_task_function(dvc_handler: Optional['DVCHandler']):
    """
    Factory function: Create a DVC version reports task function.

    Supports lazy initialization for CI/CD compatibility.

    Args:
        dvc_handler: DVCHandler instance or None for lazy init

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        # Lazy initialization at runtime
        if dvc_handler is None:
            from utils.dvc_handler import DVCHandler
            from utils.config import PipelineConfig
            config = PipelineConfig()
            handler = DVCHandler(
                repo_path="/opt/airflow",
                use_gcs=config.USE_GCS,
                gcs_bucket=config.GCS_BUCKET,
                project_id=config.PROJECT_ID
            )
        else:
            handler = dvc_handler

        return version_reports_task(
            dvc_handler=handler,
            **context
        )
    return task_function


def create_dvc_version_all_data_task_function(dvc_handler: Optional['DVCHandler']):
    """
    Factory function: Create a DVC version all data task function.

    Supports lazy initialization for CI/CD compatibility.

    Args:
        dvc_handler: DVCHandler instance or None for lazy init

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        # Lazy initialization at runtime
        if dvc_handler is None:
            from utils.dvc_handler import DVCHandler
            from utils.config import PipelineConfig
            config = PipelineConfig()
            handler = DVCHandler(
                repo_path="/opt/airflow",
                use_gcs=config.USE_GCS,
                gcs_bucket=config.GCS_BUCKET,
                project_id=config.PROJECT_ID
            )
        else:
            handler = dvc_handler

        return version_all_data_task(
            dvc_handler=handler,
            **context
        )
    return task_function


def create_dvc_version_bigquery_task_function(
    dvc_handler: Optional['DVCHandler'],
    config: 'PipelineConfig'
):
    """
    Factory function: Create a DVC version BigQuery task function.

    Supports lazy initialization for CI/CD compatibility.

    Args:
        dvc_handler: DVCHandler instance or None for lazy init
        config: PipelineConfig instance

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        # Lazy initialization at runtime
        if dvc_handler is None:
            from utils.dvc_handler import DVCHandler
            handler = DVCHandler(
                repo_path="/opt/airflow",
                use_gcs=config.USE_GCS,
                gcs_bucket=config.GCS_BUCKET,
                project_id=config.PROJECT_ID
            )
        else:
            handler = dvc_handler

        return version_bigquery_layers_task(
            dvc_handler=handler,
            config=config,
            **context
        )
    return task_function
