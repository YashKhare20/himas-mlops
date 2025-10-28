"""
HIMAS Schema & Statistics Validator
Automated schema generation, statistics computation, and quality validation
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


class SchemaValidator:
    """
    Handles schema extraction, statistics generation, and data quality validation.

    Features:
    - Extract and version BigQuery schemas
    - Generate comprehensive statistics
    - Detect schema drift over time
    - Validate data quality metrics
    - Track historical baselines
    """

    def __init__(self, project_id: str, location: str = "US"):
        """
        Initialize schema validator.

        Args:
            project_id: GCP project ID
            location: BigQuery location
        """
        self.project_id = project_id
        self.location = location
        self.client = bigquery.Client(project=project_id, location=location)
        self.logger = logging.getLogger(__name__)

    def extract_table_schema(self, dataset_id: str, table_id: str) -> Dict[str, Any]:
        """
        Extract schema from a BigQuery table.

        Args:
            dataset_id: Dataset ID
            table_id: Table ID

        Returns:
            Dictionary containing schema information
        """
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"

        try:
            table = self.client.get_table(table_ref)

            schema_dict = {
                "table_id": table_id,
                "dataset_id": dataset_id,
                "project_id": self.project_id,
                "created": table.created.isoformat() if table.created else None,
                "modified": table.modified.isoformat() if table.modified else None,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "table_type": table.table_type,
                "schema_version": datetime.now().isoformat(),
                "fields": []
            }

            # Extract field information
            for field in table.schema:
                field_dict = {
                    "name": field.name,
                    "field_type": field.field_type,
                    "mode": field.mode,
                    "description": field.description,
                }

                # Handle nested/repeated fields
                if field.fields:
                    field_dict["nested_fields"] = [
                        {
                            "name": f.name,
                            "field_type": f.field_type,
                            "mode": f.mode
                        }
                        for f in field.fields
                    ]

                schema_dict["fields"].append(field_dict)

            return schema_dict

        except NotFound:
            self.logger.error(f"Table not found: {table_ref}")
            raise
        except Exception as e:
            self.logger.error(
                f"Error extracting schema for {table_ref}: {str(e)}")
            raise

    def extract_layer_schemas(self, dataset_id: str) -> Dict[str, Dict]:
        """
        Extract schemas for all tables in a dataset/layer.

        Args:
            dataset_id: Dataset ID (e.g., 'himas_curated')

        Returns:
            Dictionary mapping table_id to schema dict
        """
        schemas = {}

        try:
            dataset_ref = f"{self.project_id}.{dataset_id}"
            tables = self.client.list_tables(dataset_ref)

            for table in tables:
                table_id = table.table_id
                self.logger.info(
                    f"Extracting schema for {dataset_id}.{table_id}")

                schema = self.extract_table_schema(dataset_id, table_id)
                schemas[table_id] = schema

            return schemas

        except NotFound:
            self.logger.error(f"Dataset not found: {dataset_id}")
            raise
        except Exception as e:
            self.logger.error(
                f"Error extracting schemas for {dataset_id}: {str(e)}")
            raise

    def compute_table_statistics(self, dataset_id: str, table_id: str) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for a BigQuery table.

        Args:
            dataset_id: Dataset ID
            table_id: Table ID

        Returns:
            Dictionary containing statistical measures
        """
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"

        try:
            # Get basic table info
            table = self.client.get_table(table_ref)

            stats = {
                "table_id": table_id,
                "dataset_id": dataset_id,
                "timestamp": datetime.now().isoformat(),
                "row_count": table.num_rows,
                "size_bytes": table.num_bytes,
                "size_mb": round(table.num_bytes / (1024 * 1024), 2),
                "field_statistics": {}
            }

            # Build query to get column-level statistics
            field_stats_queries = []

            for field in table.schema:
                field_name = field.name
                field_type = field.field_type

                # Numeric fields
                if field_type in ['INTEGER', 'INT64', 'FLOAT', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC']:
                    field_stats_queries.append(f"""
                        COUNT({field_name}) as {field_name}_count,
                        COUNT(DISTINCT {field_name}) as {field_name}_distinct,
                        AVG(CAST({field_name} AS FLOAT64)) as {field_name}_mean,
                        MIN(CAST({field_name} AS FLOAT64)) as {field_name}_min,
                        MAX(CAST({field_name} AS FLOAT64)) as {field_name}_max,
                        STDDEV(CAST({field_name} AS FLOAT64)) as {field_name}_stddev
                    """)

                # String fields
                elif field_type in ['STRING', 'TEXT']:
                    field_stats_queries.append(f"""
                        COUNT({field_name}) as {field_name}_count,
                        COUNT(DISTINCT {field_name}) as {field_name}_distinct,
                        AVG(LENGTH({field_name})) as {field_name}_avg_length
                    """)

                # Timestamp/Date fields
                elif field_type in ['TIMESTAMP', 'DATETIME', 'DATE']:
                    field_stats_queries.append(f"""
                        COUNT({field_name}) as {field_name}_count,
                        MIN({field_name}) as {field_name}_min,
                        MAX({field_name}) as {field_name}_max
                    """)

                # Boolean fields
                elif field_type in ['BOOL', 'BOOLEAN']:
                    field_stats_queries.append(f"""
                        COUNT({field_name}) as {field_name}_count,
                        COUNTIF({field_name} = TRUE) as {field_name}_true_count,
                        COUNTIF({field_name} = FALSE) as {field_name}_false_count
                    """)

            # Execute statistics query
            if field_stats_queries:
                stats_query = f"""
                    SELECT
                        {', '.join(field_stats_queries)}
                    FROM `{table_ref}`
                """

                query_job = self.client.query(stats_query)
                results = list(query_job.result())

                if results:
                    row = dict(results[0])

                    # Organize statistics by field
                    for field in table.schema:
                        field_name = field.name
                        field_stats = {}

                        # Extract relevant statistics
                        for key, value in row.items():
                            if key.startswith(f"{field_name}_"):
                                stat_name = key.replace(f"{field_name}_", "")

                                # Convert timestamps to ISO format
                                if isinstance(value, datetime):
                                    value = value.isoformat()
                                # Handle None values
                                elif value is None:
                                    value = None
                                # Round floats
                                elif isinstance(value, float):
                                    value = round(value, 4)

                                field_stats[stat_name] = value

                        if field_stats:
                            stats["field_statistics"][field_name] = {
                                "type": field.field_type,
                                "mode": field.mode,
                                **field_stats
                            }

            return stats

        except Exception as e:
            self.logger.error(
                f"Error computing statistics for {table_ref}: {str(e)}")
            raise

    def compute_layer_statistics(self, dataset_id: str) -> Dict[str, Dict]:
        """
        Compute statistics for all tables in a layer.

        Args:
            dataset_id: Dataset ID

        Returns:
            Dictionary mapping table_id to statistics
        """
        layer_stats = {}

        try:
            dataset_ref = f"{self.project_id}.{dataset_id}"
            tables = self.client.list_tables(dataset_ref)

            for table in tables:
                table_id = table.table_id
                self.logger.info(
                    f"Computing statistics for {dataset_id}.{table_id}")

                stats = self.compute_table_statistics(dataset_id, table_id)
                layer_stats[table_id] = stats

            return layer_stats

        except Exception as e:
            self.logger.error(
                f"Error computing layer statistics for {dataset_id}: {str(e)}")
            raise

    def detect_schema_drift(
        self,
        current_schema: Dict[str, Any],
        baseline_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect schema changes between current and baseline.

        Args:
            current_schema: Current table schema
            baseline_schema: Baseline schema to compare against

        Returns:
            Dictionary containing drift analysis
        """
        drift = {
            "table_id": current_schema["table_id"],
            "has_drift": False,
            "drift_detected_at": datetime.now().isoformat(),
            "changes": {
                "added_fields": [],
                "removed_fields": [],
                "modified_fields": [],
                "row_count_change": None,
                "size_change_mb": None
            }
        }

        # Build field maps
        current_fields = {f["name"]: f for f in current_schema["fields"]}
        baseline_fields = {f["name"]: f for f in baseline_schema["fields"]}

        # Detect added fields
        added = set(current_fields.keys()) - set(baseline_fields.keys())
        if added:
            drift["has_drift"] = True
            drift["changes"]["added_fields"] = list(added)

        # Detect removed fields
        removed = set(baseline_fields.keys()) - set(current_fields.keys())
        if removed:
            drift["has_drift"] = True
            drift["changes"]["removed_fields"] = list(removed)

        # Detect modified fields
        common_fields = set(current_fields.keys()) & set(
            baseline_fields.keys())
        for field_name in common_fields:
            current_field = current_fields[field_name]
            baseline_field = baseline_fields[field_name]

            # Check for type changes
            if current_field["field_type"] != baseline_field["field_type"]:
                drift["has_drift"] = True
                drift["changes"]["modified_fields"].append({
                    "field": field_name,
                    "change": "type",
                    "old": baseline_field["field_type"],
                    "new": current_field["field_type"]
                })

            # Check for mode changes
            if current_field["mode"] != baseline_field["mode"]:
                drift["has_drift"] = True
                drift["changes"]["modified_fields"].append({
                    "field": field_name,
                    "change": "mode",
                    "old": baseline_field["mode"],
                    "new": current_field["mode"]
                })

        # Row count change
        current_rows = current_schema.get("num_rows", 0)
        baseline_rows = baseline_schema.get("num_rows", 0)
        if current_rows != baseline_rows:
            drift["changes"]["row_count_change"] = {
                "baseline": baseline_rows,
                "current": current_rows,
                "delta": current_rows - baseline_rows,
                "percent_change": round(
                    ((current_rows - baseline_rows) /
                     baseline_rows * 100) if baseline_rows > 0 else 0,
                    2
                )
            }

        # Size change
        current_bytes = current_schema.get("num_bytes", 0)
        baseline_bytes = baseline_schema.get("num_bytes", 0)
        if current_bytes != baseline_bytes:
            drift["changes"]["size_change_mb"] = {
                "baseline_mb": round(baseline_bytes / (1024 * 1024), 2),
                "current_mb": round(current_bytes / (1024 * 1024), 2),
                "delta_mb": round((current_bytes - baseline_bytes) / (1024 * 1024), 2)
            }

        return drift

    def validate_data_quality(
        self,
        current_stats: Dict[str, Any],
        baseline_stats: Optional[Dict[str, Any]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Validate data quality against baseline and thresholds.

        Args:
            current_stats: Current table statistics
            baseline_stats: Baseline statistics (optional)
            thresholds: Quality thresholds (optional)

        Returns:
            Dictionary containing validation results
        """
        if thresholds is None:
            thresholds = {
                "row_count_change_pct": 50.0,  # Max 50% change
                "null_rate_threshold": 0.3,     # Max 30% nulls
                "distinct_ratio_min": 0.01      # Min 1% distinct values
            }

        validation = {
            "table_id": current_stats["table_id"],
            "validated_at": datetime.now().isoformat(),
            "passed": True,
            "warnings": [],
            "errors": [],
            "metrics": {}
        }

        # Validate row count
        current_rows = current_stats["row_count"]
        if baseline_stats:
            baseline_rows = baseline_stats["row_count"]
            if baseline_rows > 0:
                pct_change = abs(
                    (current_rows - baseline_rows) / baseline_rows * 100)

                validation["metrics"]["row_count_change_pct"] = round(
                    pct_change, 2)

                if pct_change > thresholds["row_count_change_pct"]:
                    validation["passed"] = False
                    validation["errors"].append({
                        "type": "row_count_anomaly",
                        "message": f"Row count changed by {pct_change:.2f}% (threshold: {thresholds['row_count_change_pct']}%)",
                        "baseline": baseline_rows,
                        "current": current_rows
                    })

        # Validate field-level quality
        for field_name, field_stats in current_stats.get("field_statistics", {}).items():
            field_metrics = {}

            # Null rate
            if "count" in field_stats:
                null_rate = 1 - \
                    (field_stats["count"] /
                     current_rows) if current_rows > 0 else 0
                field_metrics["null_rate"] = round(null_rate, 4)

                if null_rate > thresholds["null_rate_threshold"]:
                    validation["warnings"].append({
                        "type": "high_null_rate",
                        "field": field_name,
                        "message": f"Field has {null_rate*100:.2f}% null values (threshold: {thresholds['null_rate_threshold']*100}%)",
                        "null_rate": null_rate
                    })

            # Distinct ratio
            if "distinct" in field_stats and "count" in field_stats:
                distinct_ratio = field_stats["distinct"] / \
                    field_stats["count"] if field_stats["count"] > 0 else 0
                field_metrics["distinct_ratio"] = round(distinct_ratio, 4)

                if distinct_ratio < thresholds["distinct_ratio_min"] and field_stats["type"] in ["STRING", "TEXT"]:
                    validation["warnings"].append({
                        "type": "low_cardinality",
                        "field": field_name,
                        "message": f"Field has only {distinct_ratio*100:.2f}% distinct values",
                        "distinct_ratio": distinct_ratio
                    })

            validation["metrics"][field_name] = field_metrics

        # Set overall pass/fail
        if validation["errors"]:
            validation["passed"] = False

        return validation

    def save_schemas(self, schemas: Dict[str, Dict], output_path: Path, storage_handler=None):
        """
        Save schemas to JSON file (local or GCS based on storage_handler).

        Args:
            schemas: Dictionary of schemas to save
            output_path: Path where to save (used as key for GCS)
            storage_handler: Optional StorageHandler for GCS support
        """
        import json

        if storage_handler and storage_handler.use_gcs:
            # Save to GCS
            json_content = json.dumps(schemas, indent=2, default=str)
            blob_name = str(output_path).replace('/opt/airflow/', '')
            storage_handler.upload_string_to_gcs(json_content, blob_name)
            self.logger.info(
                f"Schemas saved to GCS: gs://{storage_handler.gcs_bucket}/{blob_name}")
        else:
            # Save locally
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(schemas, f, indent=2, default=str)
            self.logger.info(f"Schemas saved locally to {output_path}")

    def save_statistics(self, statistics: Dict[str, Dict], output_path: Path, storage_handler=None):
        """
        Save statistics to JSON file (local or GCS based on storage_handler).

        Args:
            statistics: Dictionary of statistics to save
            output_path: Path where to save (used as key for GCS)
            storage_handler: Optional StorageHandler for GCS support
        """
        import json

        if storage_handler and storage_handler.use_gcs:
            # Save to GCS
            json_content = json.dumps(statistics, indent=2, default=str)
            blob_name = str(output_path).replace('/opt/airflow/', '')
            storage_handler.upload_string_to_gcs(json_content, blob_name)
            self.logger.info(
                f"Statistics saved to GCS: gs://{storage_handler.gcs_bucket}/{blob_name}")
        else:
            # Save locally
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(statistics, f, indent=2, default=str)
            self.logger.info(f"Statistics saved locally to {output_path}")

    def save_validation(self, validation: Dict[str, Any], output_path: Path, storage_handler=None):
        """
        Save validation results to JSON file (local or GCS based on storage_handler).

        Args:
            validation: Validation results to save
            output_path: Path where to save (used as key for GCS)
            storage_handler: Optional StorageHandler for GCS support
        """
        import json

        if storage_handler and storage_handler.use_gcs:
            # Save to GCS
            json_content = json.dumps(validation, indent=2, default=str)
            blob_name = str(output_path).replace('/opt/airflow/', '')
            storage_handler.upload_string_to_gcs(json_content, blob_name)
            self.logger.info(
                f"Validation saved to GCS: gs://{storage_handler.gcs_bucket}/{blob_name}")
        else:
            # Save locally
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(validation, f, indent=2, default=str)
            self.logger.info(f"Validation saved locally to {output_path}")

    def load_baseline_schemas(self, baseline_path: Path, storage_handler=None) -> Dict[str, Dict]:
        """
        Load baseline schemas from JSON file (local or GCS based on storage_handler).

        Args:
            baseline_path: Path to baseline file
            storage_handler: Optional StorageHandler for GCS support

        Returns:
            Dictionary of baseline schemas
        """
        import json

        if storage_handler and storage_handler.use_gcs:
            # Load from GCS
            blob_name = str(baseline_path).replace('/opt/airflow/', '')
            try:
                json_content = storage_handler.download_string_from_gcs(
                    blob_name)
                return json.loads(json_content)
            except Exception as e:
                self.logger.warning(
                    f"Baseline schema file not found in GCS: {blob_name}")
                return {}
        else:
            # Load locally
            if not baseline_path.exists():
                self.logger.warning(
                    f"Baseline schema file not found: {baseline_path}")
                return {}

            with open(baseline_path, 'r') as f:
                return json.load(f)

    def load_baseline_statistics(self, baseline_path: Path, storage_handler=None) -> Dict[str, Dict]:
        """
        Load baseline statistics from JSON file (local or GCS based on storage_handler).

        Args:
            baseline_path: Path to baseline file
            storage_handler: Optional StorageHandler for GCS support

        Returns:
            Dictionary of baseline statistics
        """
        import json

        if storage_handler and storage_handler.use_gcs:
            # Load from GCS
            blob_name = str(baseline_path).replace('/opt/airflow/', '')
            try:
                json_content = storage_handler.download_string_from_gcs(
                    blob_name)
                return json.loads(json_content)
            except Exception as e:
                self.logger.warning(
                    f"Baseline statistics file not found in GCS: {blob_name}")
                return {}
        else:
            # Load locally
            if not baseline_path.exists():
                self.logger.warning(
                    f"Baseline statistics file not found: {baseline_path}")
                return {}

            with open(baseline_path, 'r') as f:
                return json.load(f)
