"""
Task Functions for HIMAS BigQuery Pipeline

Includes DVC versioning tasks and Schema & Statistics validation tasks.
"""
import logging
import json
from typing import Dict, Any
from pathlib import Path

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
# WRAPPER FUNCTIONS FOR AIRFLOW TASKS
# ============================================================================


def create_extract_schemas_task_function(schema_validator, config, storage_handler):
    """
    Factory function: Create schema extraction task function.

    Args:
        schema_validator: SchemaValidator instance
        config: PipelineConfig instance
        storage_handler: StorageHandler instance

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        return extract_all_schemas_task(
            schema_validator=schema_validator,
            config=config,
            storage_handler=storage_handler,
            **context
        )
    return task_function


def create_compute_statistics_task_function(schema_validator, config, storage_handler):
    """
    Factory function: Create statistics computation task function.

    Args:
        schema_validator: SchemaValidator instance
        config: PipelineConfig instance
        storage_handler: StorageHandler instance

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        return compute_all_statistics_task(
            schema_validator=schema_validator,
            config=config,
            storage_handler=storage_handler,
            **context
        )
    return task_function


def create_detect_drift_task_function(schema_validator, config, storage_handler):
    """
    Factory function: Create drift detection task function.

    Args:
        schema_validator: SchemaValidator instance
        config: PipelineConfig instance
        storage_handler: StorageHandler instance

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        return detect_schema_drift_task(
            schema_validator=schema_validator,
            config=config,
            storage_handler=storage_handler,
            **context
        )
    return task_function


def create_validate_quality_task_function(schema_validator, config, storage_handler):
    """
    Factory function: Create quality validation task function.

    Args:
        schema_validator: SchemaValidator instance
        config: PipelineConfig instance
        storage_handler: StorageHandler instance

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        return validate_data_quality_task(
            schema_validator=schema_validator,
            config=config,
            storage_handler=storage_handler,
            **context
        )
    return task_function


def create_quality_summary_task_function(schema_validator, config, storage_handler):
    """
    Factory function: Create quality summary task function.

    Args:
        schema_validator: SchemaValidator instance
        config: PipelineConfig instance
        storage_handler: StorageHandler instance

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        return generate_quality_summary_task(
            schema_validator=schema_validator,
            config=config,
            storage_handler=storage_handler,
            **context
        )
    return task_function


def create_dvc_version_reports_task_function(dvc_handler):
    """
    Factory function: Create a DVC version reports task function.

    Args:
        dvc_handler: DVCHandler instance

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        return version_reports_task(
            dvc_handler=dvc_handler,
            **context
        )
    return task_function


def create_dvc_version_all_data_task_function(dvc_handler):
    """
    Factory function: Create a DVC version all data task function.

    Args:
        dvc_handler: DVCHandler instance

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        return version_all_data_task(
            dvc_handler=dvc_handler,
            **context
        )
    return task_function


def create_dvc_version_bigquery_task_function(dvc_handler, config):
    """
    Factory function: Create a DVC version BigQuery task function.

    Args:
        dvc_handler: DVCHandler instance
        config: PipelineConfig instance

    Returns:
        Callable task function for Airflow
    """
    def task_function(**context):
        return version_bigquery_layers_task(
            dvc_handler=dvc_handler,
            config=config,
            **context
        )
    return task_function
