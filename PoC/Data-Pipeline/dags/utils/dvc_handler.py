"""
DVC Handler for HIMAS Data Pipeline

Manages data versioning using DVC with support for:
- Local storage
- Google Cloud Storage (GCS) remote
- Automatic version tracking
- Data integrity validation
- Dynamic reconfiguration based on USE_GCS flag
"""
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict
import json


class DVCHandler:
    """
    Handle DVC operations for data versioning.

    Supports both local and GCS-based remotes depending on USE_GCS flag.
    Automatically reconfigures remote storage if flag changes.
    """

    def __init__(
        self,
        repo_path: str = "/opt/airflow/dags",
        use_gcs: bool = False,
        gcs_bucket: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """
        Initialize DVC handler.

        Args:
            repo_path: Root path of the DVC repository
            use_gcs: Whether to use GCS as remote storage
            gcs_bucket: GCS bucket name (required if use_gcs=True)
            project_id: GCP project ID (required if use_gcs=True)
        """
        self.repo_path = Path(repo_path)
        self.use_gcs = use_gcs
        self.gcs_bucket = gcs_bucket
        self.project_id = project_id
        self.logger = logging.getLogger(__name__)

        # DVC paths
        self.dvc_dir = self.repo_path / ".dvc"
        self.dvc_config = self.dvc_dir / "config"

        # Data directories to version
        self.data_dir = self.repo_path / "data"
        self.bigquery_dir = self.data_dir / "bigquery"
        self.reports_dir = self.data_dir / "reports"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        self.models_dir = self.data_dir / "models"

    def initialize_dvc(self) -> bool:
        """
        Initialize DVC repository if not already initialized.
        Reconfigures remote if USE_GCS flag changes.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            dvc_already_exists = self.dvc_dir.exists()

            if dvc_already_exists:
                self.logger.info(
                    "DVC already initialized, checking remote configuration...")

                # Check if remote needs reconfiguration based on USE_GCS flag
                current_remote = self._get_current_remote()

                if self.use_gcs and current_remote != "gcs_storage":
                    self.logger.info(
                        "Reconfiguring DVC from local to GCS remote...")
                    self._configure_gcs_remote()
                    self.logger.info("✓ DVC reconfigured for GCS storage")
                elif not self.use_gcs and current_remote != "local_storage":
                    self.logger.info(
                        "Reconfiguring DVC from GCS to local remote...")
                    self._configure_local_remote()
                    self.logger.info("✓ DVC reconfigured for local storage")
                else:
                    self.logger.info(
                        f"✓ DVC already configured correctly for {'GCS' if self.use_gcs else 'local'} storage")

                return True

            # First time initialization
            # Check if Git is initialized, if not, initialize DVC with --no-scm
            git_dir = self.repo_path / ".git"
            use_scm = git_dir.exists()

            if use_scm:
                self.logger.info(
                    "Git repository detected, initializing DVC with SCM support...")
                init_cmd = ["dvc", "init"]
            else:
                self.logger.info(
                    "No Git repository detected, initializing DVC without SCM (--no-scm)...")
                init_cmd = ["dvc", "init", "--no-scm"]

            result = subprocess.run(
                init_cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info(f"DVC initialized: {result.stdout}")

            # Configure DVC
            self._configure_dvc()

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to initialize DVC: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error initializing DVC: {str(e)}")
            return False

    def _get_current_remote(self) -> Optional[str]:
        """
        Get the current default DVC remote name.

        Returns:
            str: Name of default remote or None
        """
        try:
            result = subprocess.run(
                ["dvc", "remote", "default"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            return None

        except Exception as e:
            self.logger.warning(
                f"Could not determine current remote: {str(e)}")
            return None

    def _configure_dvc(self):
        """Configure DVC settings and remote storage."""
        try:
            # Set autostage to automatically git add .dvc files
            subprocess.run(
                ["dvc", "config", "core.autostage", "true"],
                cwd=self.repo_path,
                check=True
            )

            # Configure remote storage based on USE_GCS flag
            if self.use_gcs:
                self._configure_gcs_remote()
                self.logger.info(
                    f"✓ DVC configured with GCS remote: gs://{self.gcs_bucket}/dvc-storage")
            else:
                self._configure_local_remote()
                self.logger.info(
                    "✓ DVC configured with local remote: .dvc_storage/")

            self.logger.info("DVC configuration completed")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to configure DVC: {e.stderr}")
            raise

    def _configure_local_remote(self):
        """Configure local remote storage for DVC."""
        local_remote_path = self.repo_path / ".dvc_storage"
        local_remote_path.mkdir(exist_ok=True)

        self.logger.info(f"Configuring local DVC remote: {local_remote_path}")

        # Remove GCS remote if it exists
        subprocess.run(
            ["dvc", "remote", "remove", "gcs_storage"],
            cwd=self.repo_path,
            capture_output=True
        )

        # Remove existing local remote if it exists
        subprocess.run(
            ["dvc", "remote", "remove", "local_storage"],
            cwd=self.repo_path,
            capture_output=True
        )

        # Add local remote as default
        subprocess.run(
            ["dvc", "remote", "add", "-d",
                "local_storage", str(local_remote_path)],
            cwd=self.repo_path,
            check=True
        )

    def _configure_gcs_remote(self):
        """Configure GCS remote storage for DVC."""
        if not self.gcs_bucket:
            raise ValueError("GCS bucket name required for GCS remote")

        gcs_url = f"gs://{self.gcs_bucket}/dvc-storage"
        self.logger.info(f"Configuring GCS DVC remote: {gcs_url}")

        # Remove local remote if it exists
        subprocess.run(
            ["dvc", "remote", "remove", "local_storage"],
            cwd=self.repo_path,
            capture_output=True
        )

        # Remove existing GCS remote if it exists
        subprocess.run(
            ["dvc", "remote", "remove", "gcs_storage"],
            cwd=self.repo_path,
            capture_output=True
        )

        # Add GCS remote as default
        subprocess.run(
            ["dvc", "remote", "add", "-d", "gcs_storage", gcs_url],
            cwd=self.repo_path,
            check=True
        )

        # Set GCS project
        if self.project_id:
            subprocess.run(
                ["dvc", "remote", "modify", "gcs_storage",
                    "projectname", self.project_id],
                cwd=self.repo_path,
                check=True
            )

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

    def create_version_metadata(self, run_id: str, context: Dict) -> Dict:
        """
        Create metadata for the current data version.

        Args:
            run_id: Airflow run ID
            context: Airflow context dictionary

        Returns:
            dict: Version metadata
        """
        # Extract information from Airflow 3.x context
        dag = context.get('dag')
        logical_date = context.get('logical_date')
        ds = context.get('ds')

        # Get timestamp (prefer logical_date, fallback to ds, then current time)
        if logical_date:
            timestamp = logical_date.strftime(
                '%Y-%m-%d %H:%M:%S') if hasattr(logical_date, 'strftime') else str(logical_date)
        elif ds:
            timestamp = ds
        else:
            from datetime import datetime
            timestamp = datetime.now().isoformat()

        # Get DAG ID
        dag_id = dag.dag_id if dag and hasattr(dag, 'dag_id') else "unknown"

        # Storage location info
        if self.use_gcs:
            storage_location = f"gs://{self.gcs_bucket}/dvc-storage/"
        else:
            storage_location = str(self.repo_path / ".dvc_storage")

        metadata = {
            "version": run_id,
            "timestamp": timestamp,
            "dag_id": dag_id,
            "run_id": run_id,
            "ds": ds,
            "use_gcs": self.use_gcs,
            "remote_type": "gcs" if self.use_gcs else "local",
            "storage_location": storage_location,
            "tracked_directories": [
                "data/bigquery",
                "data/reports",
                "data/processed",
                "data/raw",
                "data/models"
            ]
        }

        # Save metadata
        metadata_file = self.data_dir / f"version_metadata_{run_id}.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Created version metadata: {metadata_file}")
        self.logger.info(f"Storage: {storage_location}")
        return metadata

    def version_reports(self) -> bool:
        """
        Version the reports directory.

        Returns:
            bool: True if successful
        """
        try:
            # Check if reports directory exists and has files
            if not self.reports_dir.exists() or not any(self.reports_dir.iterdir()):
                self.logger.warning(
                    "Reports directory is empty or doesn't exist, skipping...")
                return True

            # Add reports directory
            success = self.add_data("data/reports")
            if not success:
                return False

            # Push to remote if USE_GCS=True
            if self.use_gcs:
                return self.push_data()

            self.logger.info("Reports versioned locally (USE_GCS=False)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to version reports: {str(e)}")
            return False

    def version_all_data(self) -> bool:
        """
        Version all data directories.

        Returns:
            bool: True if successful
        """
        try:
            data_dirs = ["data/bigquery", "data/reports",
                         "data/processed", "data/raw", "data/models"]

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

            # Push to remote if USE_GCS=True
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
        """
        Export BigQuery table to CSV for versioning.

        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            output_path: Local path to save CSV

        Returns:
            bool: True if successful
        """
        try:
            from google.cloud import bigquery

            client = bigquery.Client(project=project_id)
            table_ref = f"{project_id}.{dataset_id}.{table_id}"

            self.logger.info(f"Exporting BigQuery table: {table_ref}")

            # Query to export table
            query = f"SELECT * FROM `{table_ref}`"
            df = client.query(query).to_dataframe()

            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV locally (DVC will handle GCS if enabled)
            df.to_csv(output_file, index=False)

            self.logger.info(f"Exported {len(df)} rows to {output_file}")
            self.logger.info(
                f"Storage mode: {'Will sync to GCS via DVC' if self.use_gcs else 'Local only'}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export BigQuery table: {str(e)}")
            return False

    def version_bigquery_layers(self, project_id: str) -> bool:
        """
        Export and version all BigQuery layers.
        Data is always exported locally first, then synced to GCS if USE_GCS=True.

        Args:
            project_id: GCP project ID

        Returns:
            bool: True if successful
        """
        try:
            # Define layers and tables to export
            layers = {
                'curated': [
                    'patient_split_assignment',
                    'dim_patient',
                    'fact_hospital_admission',
                    'fact_icu_stay',
                    'fact_transfers',
                    'clinical_features'
                ],
                'federated': [
                    'hospital_a_data',
                    'hospital_b_data',
                    'hospital_c_data'
                ],
                'verification': [
                    'data_leakage_check',
                    'dataset_statistics'
                ]
            }

            self.logger.info("=" * 60)
            self.logger.info(f"BigQuery Export & Versioning")
            self.logger.info(
                f"Storage Mode: {'GCS (via DVC)' if self.use_gcs else 'Local Only'}")
            if self.use_gcs:
                self.logger.info(
                    f"GCS Location: gs://{self.gcs_bucket}/dvc-storage/")
            else:
                self.logger.info(
                    f"Local Location: {self.repo_path}/.dvc_storage/")
            self.logger.info("=" * 60)

            # Export each layer to local storage
            for layer, tables in layers.items():
                layer_dir = self.bigquery_dir / layer
                layer_dir.mkdir(parents=True, exist_ok=True)

                for table in tables:
                    output_path = layer_dir / f"{table}.csv"
                    dataset_id = f"{layer}_demo"

                    success = self.export_bigquery_to_csv(
                        project_id=project_id,
                        dataset_id=dataset_id,
                        table_id=table,
                        output_path=str(output_path)
                    )

                    if not success:
                        self.logger.warning(
                            f"Failed to export {layer}.{table}")

            # Add entire bigquery directory to DVC tracking
            self.logger.info("Adding BigQuery exports to DVC tracking...")
            if not self.add_data("data/bigquery"):
                return False

            # Push to GCS if USE_GCS=True, otherwise keep local
            if self.use_gcs:
                self.logger.info("Pushing BigQuery data to GCS via DVC...")
                if not self.push_data():
                    self.logger.warning(
                        "Failed to push to GCS, but local version saved")
                    return False
                self.logger.info("✓ BigQuery data versioned and synced to GCS")
            else:
                self.logger.info(
                    "✓ BigQuery data versioned locally in .dvc_storage/")

            return True

        except Exception as e:
            self.logger.error(f"Failed to version BigQuery layers: {str(e)}")
            return False

    def get_dvc_info(self) -> Dict:
        """
        Get DVC configuration information.

        Returns:
            dict: DVC configuration details
        """
        storage_location = f"gs://{self.gcs_bucket}/dvc-storage/" if self.use_gcs else str(
            self.repo_path / ".dvc_storage")

        return {
            "initialized": self.dvc_dir.exists(),
            "use_gcs": self.use_gcs,
            "gcs_bucket": self.gcs_bucket if self.use_gcs else None,
            "project_id": self.project_id if self.use_gcs else None,
            "remote_type": "gcs" if self.use_gcs else "local",
            "storage_location": storage_location,
            "data_directories": [
                str(self.bigquery_dir),
                str(self.reports_dir),
                str(self.processed_dir),
                str(self.raw_dir),
                str(self.models_dir)
            ]
        }
