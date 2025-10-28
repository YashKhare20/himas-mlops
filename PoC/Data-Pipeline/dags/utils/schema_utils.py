"""
Schema Utilities - Helper functions for schema validation with config integration
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from utils.config import PipelineConfig
from utils.schema_validator import SchemaValidator


logger = logging.getLogger(__name__)


def get_all_layers_config() -> List[Tuple[str, str]]:
    """
    Get all layer configurations for schema extraction/statistics.

    Returns:
        List of tuples: (dataset_id, layer_name)
    """
    return [
        (PipelineConfig.LAYERS['curated']['dataset_id'], 'curated'),
        (PipelineConfig.LAYERS['federated']['dataset_id'], 'federated'),
        (PipelineConfig.LAYERS['verification']['dataset_id'], 'verification')
    ]


def extract_all_layer_schemas(
    schema_validator: SchemaValidator,
    run_id: str,
    storage_handler=None,
    output_dir: Path = Path("/opt/airflow/data/schemas")
) -> Dict[str, Any]:
    """
    Extract schemas from all configured layers.

    Args:
        schema_validator: SchemaValidator instance
        run_id: Pipeline run ID
        storage_handler: Optional StorageHandler for GCS support
        output_dir: Directory to save schemas (used for local or as GCS key prefix)

    Returns:
        Dictionary with extraction results
    """
    # Only create directory if using local storage
    if not (storage_handler and storage_handler.use_gcs):
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting schemas for run_id: {run_id}")

    layers = get_all_layers_config()
    all_schemas = {}

    for dataset_id, layer_name in layers:
        logger.info(
            f"Extracting schema for layer: {layer_name} (dataset: {dataset_id})")

        try:
            layer_schemas = schema_validator.extract_layer_schemas(dataset_id)

            # Save layer-specific schema
            output_path = output_dir / f"schema_{layer_name}.json"
            schema_validator.save_schemas(
                layer_schemas, output_path, storage_handler)

            all_schemas[layer_name] = layer_schemas

            logger.info(
                f"Extracted {len(layer_schemas)} table schemas from {layer_name}")

        except Exception as e:
            logger.error(
                f"Failed to extract schema for {layer_name}: {str(e)}")
            raise

    # Save combined schemas
    combined_path = output_dir / f"schemas_all_{run_id}.json"
    schema_validator.save_schemas(all_schemas, combined_path, storage_handler)

    logger.info(f"All schemas extracted and saved")

    return {
        "success": True,
        "run_id": run_id,
        "schemas_dir": str(output_dir),
        "storage": "gcs" if (storage_handler and storage_handler.use_gcs) else "local",
        "layers": [layer for _, layer in layers],
        "total_tables": sum(len(schemas) for schemas in all_schemas.values()),
        "layer_details": {
            layer: {
                "dataset_id": dataset_id,
                "table_count": len(all_schemas.get(layer, {})),
                "tables": list(all_schemas.get(layer, {}).keys())
            }
            for dataset_id, layer in layers
        }
    }


def compute_all_layer_statistics(
    schema_validator: SchemaValidator,
    run_id: str,
    storage_handler=None,
    output_dir: Path = Path("/opt/airflow/data/statistics")
) -> Dict[str, Any]:
    """
    Compute statistics for all configured layers.

    Args:
        schema_validator: SchemaValidator instance
        run_id: Pipeline run ID
        storage_handler: Optional StorageHandler for GCS support
        output_dir: Directory to save statistics (used for local or as GCS key prefix)

    Returns:
        Dictionary with computation results
    """
    # Only create directory if using local storage
    if not (storage_handler and storage_handler.use_gcs):
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing statistics for run_id: {run_id}")

    layers = get_all_layers_config()
    all_statistics = {}

    for dataset_id, layer_name in layers:
        logger.info(
            f"Computing statistics for layer: {layer_name} (dataset: {dataset_id})")

        try:
            layer_stats = schema_validator.compute_layer_statistics(dataset_id)

            # Save layer-specific statistics
            output_path = output_dir / f"statistics_{layer_name}.json"
            schema_validator.save_statistics(
                layer_stats, output_path, storage_handler)

            all_statistics[layer_name] = layer_stats

            logger.info(
                f"Computed statistics for {len(layer_stats)} tables in {layer_name}")

        except Exception as e:
            logger.error(
                f"Failed to compute statistics for {layer_name}: {str(e)}")
            raise

    # Save combined statistics
    combined_path = output_dir / f"statistics_all_{run_id}.json"
    schema_validator.save_statistics(
        all_statistics, combined_path, storage_handler)

    logger.info(f"All statistics computed and saved")

    # Calculate totals
    total_rows = sum(
        table_stats["row_count"]
        for layer in all_statistics.values()
        for table_stats in layer.values()
    )

    total_size_mb = sum(
        table_stats["size_mb"]
        for layer in all_statistics.values()
        for table_stats in layer.values()
    )

    return {
        "success": True,
        "run_id": run_id,
        "stats_dir": str(output_dir),
        "storage": "gcs" if (storage_handler and storage_handler.use_gcs) else "local",
        "layers": [layer for _, layer in layers],
        "total_tables": sum(len(stats) for stats in all_statistics.values()),
        "total_rows": total_rows,
        "total_size_mb": round(total_size_mb, 2),
        "layer_details": {
            layer: {
                "dataset_id": dataset_id,
                "table_count": len(all_statistics.get(layer, {})),
                "tables": list(all_statistics.get(layer, {}).keys()),
                "row_count": sum(
                    t["row_count"] for t in all_statistics.get(layer, {}).values()
                ),
                "size_mb": round(sum(
                    t["size_mb"] for t in all_statistics.get(layer, {}).values()
                ), 2)
            }
            for dataset_id, layer in layers
        }
    }


def detect_schema_drift_all_layers(
    schema_validator: SchemaValidator,
    run_id: str,
    storage_handler=None,
    schemas_dir: Path = Path("/opt/airflow/data/schemas"),
    output_dir: Path = Path("/opt/airflow/data/drift")
) -> Dict[str, Any]:
    """
    Detect schema drift across all layers.

    Args:
        schema_validator: SchemaValidator instance
        run_id: Pipeline run ID
        storage_handler: Optional StorageHandler for GCS support
        schemas_dir: Directory containing schemas
        output_dir: Directory to save drift reports

    Returns:
        Dictionary with drift detection results
    """
    import json
    from datetime import datetime

    # Only create directory if using local storage
    if not (storage_handler and storage_handler.use_gcs):
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Detecting schema drift for run_id: {run_id}")

    # Load current schemas
    current_schemas_path = schemas_dir / f"schemas_all_{run_id}.json"

    if storage_handler and storage_handler.use_gcs:
        # Load from GCS
        blob_name = str(current_schemas_path).replace('/opt/airflow/', '')
        try:
            json_content = storage_handler.download_string_from_gcs(blob_name)
            current_schemas = json.loads(json_content)
        except Exception as e:
            logger.error(f"Current schemas not found in GCS: {blob_name}")
            raise FileNotFoundError(
                f"Current schemas not found in GCS: {blob_name}")
    else:
        # Load locally
        if not current_schemas_path.exists():
            logger.error(f"Current schemas not found: {current_schemas_path}")
            raise FileNotFoundError(
                f"Current schemas not found: {current_schemas_path}")

        with open(current_schemas_path, 'r') as f:
            current_schemas = json.load(f)

    # Load baseline schemas
    baseline_path = schemas_dir / "schemas_baseline.json"

    drift_results = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "has_baseline": False,
        "layers": {}
    }

    # Check if baseline exists
    baseline_exists = False
    if storage_handler and storage_handler.use_gcs:
        blob_name = str(baseline_path).replace('/opt/airflow/', '')
        try:
            storage_handler.download_string_from_gcs(blob_name)
            baseline_exists = True
        except:
            baseline_exists = False
    else:
        baseline_exists = baseline_path.exists()

    drift_results["has_baseline"] = baseline_exists

    if baseline_exists:
        logger.info(f"Loading baseline schemas")
        baseline_schemas = schema_validator.load_baseline_schemas(
            baseline_path, storage_handler)

        # Compare each layer
        for layer_name, layer_tables in current_schemas.items():
            layer_drift = {}

            if layer_name in baseline_schemas:
                baseline_tables = baseline_schemas[layer_name]

                # Compare each table
                for table_id, current_schema in layer_tables.items():
                    if table_id in baseline_tables:
                        drift = schema_validator.detect_schema_drift(
                            current_schema,
                            baseline_tables[table_id]
                        )
                        layer_drift[table_id] = drift
                    else:
                        # New table
                        layer_drift[table_id] = {
                            "table_id": table_id,
                            "has_drift": True,
                            "changes": {"new_table": True}
                        }
            else:
                # New layer
                logger.info(f"New layer detected: {layer_name}")
                for table_id in layer_tables.keys():
                    layer_drift[table_id] = {
                        "table_id": table_id,
                        "has_drift": True,
                        "changes": {"new_layer": True}
                    }

            drift_results["layers"][layer_name] = layer_drift

        # Count total drifts
        total_drifts = sum(
            1 for layer in drift_results["layers"].values()
            for table_drift in layer.values()
            if table_drift.get("has_drift", False)
        )
        drift_results["total_drifts"] = total_drifts

        logger.info(f"Detected {total_drifts} schema drifts")
    else:
        logger.info("No baseline found - establishing baseline")
        # Save current as baseline
        schema_validator.save_schemas(
            current_schemas, baseline_path, storage_handler)
        drift_results["message"] = "Baseline established"

    # Save drift report
    drift_report_path = output_dir / f"schema_drift_{run_id}.json"
    schema_validator.save_validation(
        drift_results, drift_report_path, storage_handler)

    logger.info(f"Schema drift report saved")

    return drift_results


def validate_data_quality_all_layers(
    schema_validator: SchemaValidator,
    run_id: str,
    storage_handler=None,
    stats_dir: Path = Path("/opt/airflow/data/statistics"),
    output_dir: Path = Path("/opt/airflow/data/validation"),
    custom_thresholds: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Validate data quality across all layers.

    Args:
        schema_validator: SchemaValidator instance
        run_id: Pipeline run ID
        storage_handler: Optional StorageHandler for GCS support
        stats_dir: Directory containing statistics
        output_dir: Directory to save validation reports
        custom_thresholds: Optional custom quality thresholds

    Returns:
        Dictionary with validation results
    """
    import json
    from datetime import datetime

    # Only create directory if using local storage
    if not (storage_handler and storage_handler.use_gcs):
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Validating data quality for run_id: {run_id}")

    # Load current statistics
    current_stats_path = stats_dir / f"statistics_all_{run_id}.json"

    if storage_handler and storage_handler.use_gcs:
        # Load from GCS
        blob_name = str(current_stats_path).replace('/opt/airflow/', '')
        try:
            json_content = storage_handler.download_string_from_gcs(blob_name)
            current_statistics = json.loads(json_content)
        except Exception as e:
            logger.error(f"Current statistics not found in GCS: {blob_name}")
            raise FileNotFoundError(
                f"Current statistics not found in GCS: {blob_name}")
    else:
        # Load locally
        if not current_stats_path.exists():
            logger.error(f"Current statistics not found: {current_stats_path}")
            raise FileNotFoundError(
                f"Current statistics not found: {current_stats_path}")

        with open(current_stats_path, 'r') as f:
            current_statistics = json.load(f)

    # Load baseline statistics
    baseline_path = stats_dir / "statistics_baseline.json"

    validation_results = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "has_baseline": False,
        "overall_passed": True,
        "layers": {}
    }

    # Check if baseline exists
    baseline_exists = False
    if storage_handler and storage_handler.use_gcs:
        blob_name = str(baseline_path).replace('/opt/airflow/', '')
        try:
            storage_handler.download_string_from_gcs(blob_name)
            baseline_exists = True
        except:
            baseline_exists = False
    else:
        baseline_exists = baseline_path.exists()

    validation_results["has_baseline"] = baseline_exists

    if baseline_exists:
        logger.info(f"Loading baseline statistics")
        baseline_statistics = schema_validator.load_baseline_statistics(
            baseline_path, storage_handler)
    else:
        logger.info("No baseline found - establishing baseline")
        baseline_statistics = {}
        # Save current as baseline
        schema_validator.save_statistics(
            current_statistics, baseline_path, storage_handler)

    # Validate each layer
    for layer_name, layer_tables in current_statistics.items():
        layer_validations = {}

        baseline_layer = baseline_statistics.get(layer_name, {})

        for table_id, current_stats in layer_tables.items():
            baseline_stats = baseline_layer.get(table_id)

            validation = schema_validator.validate_data_quality(
                current_stats,
                baseline_stats,
                thresholds=custom_thresholds
            )

            layer_validations[table_id] = validation

            if not validation["passed"]:
                validation_results["overall_passed"] = False

        validation_results["layers"][layer_name] = layer_validations

    # Count failures and warnings
    total_errors = sum(
        len(table_val.get("errors", []))
        for layer in validation_results["layers"].values()
        for table_val in layer.values()
    )

    total_warnings = sum(
        len(table_val.get("warnings", []))
        for layer in validation_results["layers"].values()
        for table_val in layer.values()
    )

    validation_results["total_errors"] = total_errors
    validation_results["total_warnings"] = total_warnings

    logger.info(
        f"Quality validation: {total_errors} errors, {total_warnings} warnings")

    # Save validation report
    validation_report_path = output_dir / f"quality_validation_{run_id}.json"
    schema_validator.save_validation(
        validation_results, validation_report_path, storage_handler)

    logger.info(f"Quality validation report saved")

    # Raise exception if validation failed
    if not validation_results["overall_passed"]:
        error_msg = f"Data quality validation failed with {total_errors} errors"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return validation_results


def generate_comprehensive_quality_summary(
    run_id: str,
    config: PipelineConfig = None,
    storage_handler=None
) -> Dict[str, Any]:
    """
    Generate comprehensive quality summary combining all reports.

    Args:
        run_id: Pipeline run ID
        config: Pipeline configuration
        storage_handler: Optional StorageHandler for GCS support

    Returns:
        Dictionary with comprehensive summary
    """
    import json
    from datetime import datetime

    logger.info(f"Generating quality summary for run_id: {run_id}")

    if config is None:
        config = PipelineConfig()

    # Define paths
    schemas_path = Path("/opt/airflow/data/schemas") / \
        f"schemas_all_{run_id}.json"
    stats_path = Path("/opt/airflow/data/statistics") / \
        f"statistics_all_{run_id}.json"
    drift_path = Path("/opt/airflow/data/drift") / \
        f"schema_drift_{run_id}.json"
    validation_path = Path("/opt/airflow/data/validation") / \
        f"quality_validation_{run_id}.json"

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "project_id": config.PROJECT_ID,
        "pipeline": "HIMAS BigQuery Pipeline",
        "storage": "gcs" if (storage_handler and storage_handler.use_gcs) else "local",
        "summary": {},
        "layer_breakdown": {}
    }

    # Helper function to load JSON from GCS or local
    def load_json(path: Path) -> dict:
        if storage_handler and storage_handler.use_gcs:
            blob_name = str(path).replace('/opt/airflow/', '')
            try:
                json_content = storage_handler.download_string_from_gcs(
                    blob_name)
                return json.loads(json_content)
            except:
                return None
        else:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
            return None

    # Load and summarize schemas
    schemas = load_json(schemas_path)
    if schemas:
        summary["summary"]["schemas"] = {
            "total_layers": len(schemas),
            "total_tables": sum(len(tables) for tables in schemas.values()),
            "layers": {
                layer: {
                    "table_count": len(tables),
                    "tables": list(tables.keys())
                }
                for layer, tables in schemas.items()
            }
        }

    # Load and summarize statistics
    statistics = load_json(stats_path)
    if statistics:
        summary["summary"]["statistics"] = {
            "total_rows": sum(
                table_stats["row_count"]
                for layer in statistics.values()
                for table_stats in layer.values()
            ),
            "total_size_mb": round(sum(
                table_stats["size_mb"]
                for layer in statistics.values()
                for table_stats in layer.values()
            ), 2),
            "by_layer": {
                layer: {
                    "rows": sum(t["row_count"] for t in tables.values()),
                    "size_mb": round(sum(t["size_mb"] for t in tables.values()), 2)
                }
                for layer, tables in statistics.items()
            }
        }

    # Load and summarize drift
    drift = load_json(drift_path)
    if drift:
        summary["summary"]["drift"] = {
            "has_baseline": drift.get("has_baseline", False),
            "total_drifts": drift.get("total_drifts", 0),
            "by_layer": {
                layer: sum(1 for t in tables.values()
                           if t.get("has_drift", False))
                for layer, tables in drift.get("layers", {}).items()
            }
        }

    # Load and summarize validation
    validation = load_json(validation_path)
    if validation:
        summary["summary"]["validation"] = {
            "overall_passed": validation.get("overall_passed", False),
            "total_errors": validation.get("total_errors", 0),
            "total_warnings": validation.get("total_warnings", 0),
            "by_layer": {
                layer: {
                    "passed": all(t.get("passed", False) for t in tables.values()),
                    "errors": sum(len(t.get("errors", [])) for t in tables.values()),
                    "warnings": sum(len(t.get("warnings", [])) for t in tables.values())
                }
                for layer, tables in validation.get("layers", {}).items()
            }
        }

    # Save summary
    summary_path = Path("/opt/airflow/data/reports") / \
        f"quality_summary_{run_id}.json"

    if storage_handler and storage_handler.use_gcs:
        # Save to GCS
        json_content = json.dumps(summary, indent=2)
        blob_name = str(summary_path).replace('/opt/airflow/', '')
        storage_handler.upload_string_to_gcs(json_content, blob_name)
        logger.info(
            f"Quality summary saved to GCS: gs://{storage_handler.gcs_bucket}/{blob_name}")
    else:
        # Save locally
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Quality summary saved locally to {summary_path}")

    return summary
