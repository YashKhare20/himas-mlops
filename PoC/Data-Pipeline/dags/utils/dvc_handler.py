"""
DVC Handler for HIMAS Data Pipeline

Manages data versioning using DVC with support for:
- Works with DVC initialized at project root
- Local storage with Git versioning
- Google Cloud Storage (GCS) remote
- Automatic version tracking
- Metadata versioning with local/GCS storage
"""
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict
import json


class DVCHandler:
    """
    Handle DVC operations for data versioning.

    Assumes DVC is already initialized at project root.
    Supports both local and GCS-based remotes depending on USE_GCS flag.
    """

    def __init__(
        self,
        repo_path: str = "/opt/airflow",
        use_gcs: bool = False,
        gcs_bucket: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """
        Initialize DVC handler.

        Args:
            repo_path: Root path of the DVC repository (project root, not dags/)
            use_gcs: Whether to use GCS as remote storage
            gcs_bucket: GCS bucket name (required if use_gcs=True)
            project_id: GCP project ID (required if use_gcs=True)
        """
        self.repo_path = Path(repo_path)
        self.use_gcs = use_gcs
        self.gcs_bucket = gcs_bucket
        self.project_id = project_id
        self.logger = logging.getLogger(__name__)

        # DVC paths (at project root)
        self.dvc_dir = self.repo_path / ".dvc"
        self.git_dir = self.repo_path / ".git"

        # Data directories to version (inside dags/)
        self.dags_dir = self.repo_path / "dags"
        self.data_dir = self.dags_dir / "data"
        self.bigquery_dir = self.data_dir / "bigquery"
        self.reports_dir = self.data_dir / "reports"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        self.models_dir = self.data_dir / "models"
        self.metadata_dir = self.data_dir / "metadata"

    def verify_dvc_initialized(self) -> bool:
        """
        Verify that DVC is initialized at project root.

        Returns:
            bool: True if DVC is initialized, False otherwise
        """
        if not self.dvc_dir.exists():
            self.logger.error(f"DVC not initialized at {self.repo_path}")
            self.logger.error("Run: scripts/init_dvc_root.sh")
            return False

        if not self.git_dir.exists():
            self.logger.warning(f"Git not initialized at {self.repo_path}")

        self.logger.info(
            f"✓ DVC initialized at project root: {self.repo_path}")
        return True

    def _git_commit(self, message: str, files: List[str] = None):
        """
        Commit changes to Git.

        Args:
            message: Commit message
            files: List of files/directories to add (None = add all)
        """
        try:
            if not self.git_dir.exists():
                return

            # Add files
            if files:
                for file in files:
                    subprocess.run(
                        ["git", "add", file],
                        cwd=self.repo_path,
                        capture_output=True,
                        check=False
                    )
            else:
                subprocess.run(
                    ["git", "add", "."],
                    cwd=self.repo_path,
                    capture_output=True,
                    check=False
                )

            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                capture_output=True,
                check=False
            )

            if result.returncode == 0:
                self.logger.info(f"✓ Git commit: {message}")

        except Exception as e:
            self.logger.warning(f"Git commit failed: {str(e)}")

    def add_data(self, data_path: str) -> bool:
        """
        Add data file or directory to DVC tracking.

        Args:
            data_path: Path to data file or directory (relative to repo_path)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            full_path = self.repo_path / data_path

            if not full_path.exists():
                self.logger.error(f"Data path does not exist: {full_path}")
                return False

            self.logger.info(f"Adding {data_path} to DVC tracking...")

            result = subprocess.run(
                ["dvc", "add", str(data_path)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            self.logger.info(f"Successfully added to DVC: {result.stdout}")

            # Commit .dvc file to Git
            dvc_file = f"{data_path}.dvc"
            if self.git_dir.exists():
                self._git_commit(f"Track {data_path} with DVC", [
                                 dvc_file, ".gitignore"])

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to add data to DVC: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error adding data: {str(e)}")
            return False

    def push_data(self) -> bool:
        """
        Push DVC-tracked data to remote storage.
        Only pushes if USE_GCS=True.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.use_gcs:
                self.logger.info(
                    "Skipping push - USE_GCS=False (data stored locally)")
                return True

            self.logger.info("Pushing data to GCS via DVC...")

            result = subprocess.run(
                ["dvc", "push"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            self.logger.info(f"Successfully pushed to GCS: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to push data to GCS: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error pushing data: {str(e)}")
            return False

    def pull_data(self) -> bool:
        """
        Pull DVC-tracked data from remote storage.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(
                f"Pulling data from {'GCS' if self.use_gcs else 'local'} remote...")

            result = subprocess.run(
                ["dvc", "pull"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            self.logger.info(
                f"Successfully pulled from remote: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to pull data: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error pulling data: {str(e)}")
            return False

    def get_data_status(self) -> Dict:
        """
        Get status of DVC-tracked data.

        Returns:
            dict: Status information about tracked data
        """
        try:
            result = subprocess.run(
                ["dvc", "status"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            return {
                "status": "success",
                "output": result.stdout,
                "message": "Data is up to date" if not result.stdout else "Changes detected",
                "storage_type": "gcs" if self.use_gcs else "local"
            }

        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "output": e.stderr,
                "message": "Failed to get DVC status",
                "storage_type": "gcs" if self.use_gcs else "local"
            }

    def save_metadata_to_storage(self, metadata: Dict, filename: str) -> str:
        """
        Save metadata to local file or GCS based on USE_GCS flag.

        Args:
            metadata: Metadata dictionary to save
            filename: Name of the metadata file

        Returns:
            str: Path or URI where metadata was saved
        """
        try:
            # Ensure metadata directory exists
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

            if self.use_gcs:
                # Save to GCS
                from google.cloud import storage

                client = storage.Client(project=self.project_id)
                bucket = client.bucket(self.gcs_bucket)
                blob = bucket.blob(f"metadata/{filename}")

                # Upload metadata as JSON
                blob.upload_from_string(
                    json.dumps(metadata, indent=2),
                    content_type='application/json'
                )

                gcs_uri = f"gs://{self.gcs_bucket}/metadata/{filename}"
                self.logger.info(f"✓ Metadata saved to GCS: {gcs_uri}")
                return gcs_uri

            else:
                # Save locally
                local_path = self.metadata_dir / filename
                with open(local_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                self.logger.info(f"✓ Metadata saved locally: {local_path}")
                return str(local_path)

        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")
            # Fallback to local save
            local_path = self.metadata_dir / filename
            with open(local_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            return str(local_path)

    def create_version_metadata(self, run_id: str, context: Dict) -> Dict:
        """
        Create metadata for the current data version.
        Saves to dags/data/metadata/ locally or GCS based on USE_GCS flag.

        Args:
            run_id: Airflow run ID
            context: Airflow context dictionary

        Returns:
            dict: Version metadata
        """
        # Extract information from Airflow 3.x context
        dag = context.get('dag')
        logical_date = context.get('logical_date')
        timestamp = logical_date.strftime(
                '%Y-%m-%d %H:%M:%S') if hasattr(logical_date, 'strftime') else str(logical_date)

        # Get DAG ID
        dag_id = dag.dag_id if dag and hasattr(dag, 'dag_id') else "unknown"

        # Storage location info
        if self.use_gcs:
            storage_location = f"gs://{self.gcs_bucket}/dvc-storage/"
            metadata_location = f"gs://{self.gcs_bucket}/metadata/"
        else:
            storage_location = str(self.dvc_dir / "cache")
            metadata_location = str(self.metadata_dir)

        metadata = {
            "version": run_id,
            "timestamp": timestamp,
            "dag_id": dag_id,
            "run_id": run_id,
            "use_gcs": self.use_gcs,
            "remote_type": "gcs" if self.use_gcs else "local",
            "storage_location": storage_location,
            "metadata_location": metadata_location,
            "git_enabled": self.git_dir.exists(),
            "dvc_root": str(self.repo_path),
            "tracked_directories": [
                "dags/data/bigquery",
                "dags/data/reports",
                "dags/data/processed",
                "dags/data/raw",
                "dags/data/models"
            ]
        }

        # Save metadata using storage handler
        filename = f"version_metadata_{run_id}.json"
        saved_location = self.save_metadata_to_storage(metadata, filename)

        metadata["metadata_file"] = saved_location
        self.logger.info(f"Created version metadata at: {saved_location}")

        return metadata

    def version_reports(self) -> bool:
        """Version the reports directory."""
        try:
            if not self.reports_dir.exists() or not any(self.reports_dir.iterdir()):
                self.logger.warning(
                    "Reports directory is empty or doesn't exist, skipping...")
                return True

            success = self.add_data("dags/data/reports")
            if not success:
                return False

            if self.use_gcs:
                return self.push_data()

            self.logger.info("Reports versioned locally (USE_GCS=False)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to version reports: {str(e)}")
            return False

    def version_all_data(self) -> bool:
        """Version all data directories."""
        try:
            data_dirs = ["dags/data/bigquery", "dags/data/reports",
                         "dags/data/processed", "dags/data/raw", "dags/data/models"]

            for data_dir in data_dirs:
                dir_path = self.repo_path / data_dir
                if dir_path.exists() and any(dir_path.iterdir()):
                    self.logger.info(f"Versioning {data_dir}...")
                    if not self.add_data(data_dir):
                        self.logger.warning(
                            f"Failed to version {data_dir}, continuing...")
                else:
                    self.logger.info(
                        f"Skipping {data_dir} (empty or doesn't exist)")

            if self.use_gcs:
                self.logger.info("Pushing all data to GCS...")
                return self.push_data()
            else:
                self.logger.info("All data versioned locally (USE_GCS=False)")
                return True

        except Exception as e:
            self.logger.error(f"Failed to version all data: {str(e)}")
            return False

    def export_bigquery_to_csv(self, project_id: str, dataset_id: str, table_id: str, output_path: str) -> bool:
        """Export BigQuery table to CSV for versioning."""
        try:
            from google.cloud import bigquery

            client = bigquery.Client(project=project_id)
            table_ref = f"{project_id}.{dataset_id}.{table_id}"

            self.logger.info(f"Exporting BigQuery table: {table_ref}")

            query = f"SELECT * FROM `{table_ref}`"
            df = client.query(query).to_dataframe()

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(output_file, index=False)

            self.logger.info(f"Exported {len(df)} rows to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export BigQuery table: {str(e)}")
            return False

    def version_bigquery_layers(self, project_id: str) -> bool:
        """Export and version all BigQuery layers."""
        try:
            layers = {
                'curated': ['patient_split_assignment', 'dim_patient', 'fact_hospital_admission', 'fact_icu_stay', 'fact_transfers', 'clinical_features'],
                'federated': ['hospital_a_data', 'hospital_b_data', 'hospital_c_data'],
                'verification': ['data_leakage_check', 'dataset_statistics']
            }

            self.logger.info("=" * 60)
            self.logger.info(f"BigQuery Export & Versioning")
            self.logger.info(f"DVC Root: {self.repo_path}")
            self.logger.info(
                f"Storage Mode: {'GCS' if self.use_gcs else 'Local'}")
            self.logger.info("=" * 60)

            for layer, tables in layers.items():
                layer_dir = self.bigquery_dir / layer
                layer_dir.mkdir(parents=True, exist_ok=True)

                for table in tables:
                    output_path = layer_dir / f"{table}.csv"
                    dataset_id = f"{layer}_demo"

                    success = self.export_bigquery_to_csv(
                        project_id, dataset_id, table, str(output_path))
                    if not success:
                        self.logger.warning(
                            f"Failed to export {layer}.{table}")

            self.logger.info("Adding BigQuery exports to DVC tracking...")
            if not self.add_data("dags/data/bigquery"):
                return False

            if self.use_gcs:
                self.logger.info("Pushing BigQuery data to GCS via DVC...")
                if not self.push_data():
                    self.logger.warning(
                        "Failed to push to GCS, but local version saved")
                    return False
                self.logger.info("✓ BigQuery data versioned and synced to GCS")
            else:
                self.logger.info("✓ BigQuery data versioned locally")

            return True

        except Exception as e:
            self.logger.error(f"Failed to version BigQuery layers: {str(e)}")
            return False

    def get_dvc_info(self) -> Dict:
        """Get DVC configuration information."""
        storage_location = f"gs://{self.gcs_bucket}/dvc-storage/" if self.use_gcs else str(
            self.dvc_dir / "cache")
        metadata_location = f"gs://{self.gcs_bucket}/metadata/" if self.use_gcs else str(
            self.metadata_dir)

        return {
            "initialized": self.dvc_dir.exists(),
            "dvc_root": str(self.repo_path),
            "git_enabled": self.git_dir.exists(),
            "use_gcs": self.use_gcs,
            "gcs_bucket": self.gcs_bucket if self.use_gcs else None,
            "project_id": self.project_id if self.use_gcs else None,
            "remote_type": "gcs" if self.use_gcs else "local",
            "storage_location": storage_location,
            "metadata_location": metadata_location
        }
