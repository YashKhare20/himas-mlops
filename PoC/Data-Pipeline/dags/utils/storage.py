"""
Storage utilities for HIMAS Pipeline.

Provides hybrid storage support (local filesystem or GCS).
"""
import logging
from pathlib import Path
from typing import Optional


class StorageHandler:
    """Handles storage operations for local or GCS based on configuration."""

    def __init__(self, use_gcs: bool, gcs_bucket: Optional[str] = None,
                 local_dir: Optional[Path] = None, project_id: Optional[str] = None):
        """
        Initialize storage handler.

        Args:
            use_gcs: Whether to use GCS
            gcs_bucket: GCS bucket name (required if use_gcs=True)
            local_dir: Local directory for storage
            project_id: GCP project ID (required if use_gcs=True)
        """
        self.use_gcs = use_gcs
        self.gcs_bucket = gcs_bucket
        self.local_dir = local_dir or Path('/opt/airflow/dags/data/reports')
        self.project_id = project_id

        # Ensure local directory exists
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: str, filename: str, content_type: str = 'text/csv') -> str:
        """
        Save data to configured storage.

        Args:
            data: String data to save
            filename: Filename (e.g., 'statistics.csv')
            content_type: MIME type

        Returns:
            URI/path where file was saved
        """
        if self.use_gcs:
            try:
                return self._save_to_gcs(data, filename, content_type)
            except Exception as e:
                logging.warning(f"GCS save failed, falling back to local: {e}")
                return self._save_to_local(data, filename)
        else:
            return self._save_to_local(data, filename)

    def _save_to_gcs(self, data: str, filename: str, content_type: str) -> str:
        """Save data to Google Cloud Storage."""
        from google.cloud import storage

        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.gcs_bucket)
        blob = bucket.blob(f'reports/{filename}')
        blob.upload_from_string(data, content_type=content_type)

        location = f"gs://{self.gcs_bucket}/reports/{filename}"
        logging.info(f"Saved to GCS: {location}")
        return location

    def _save_to_local(self, data: str, filename: str) -> str:
        """Save data to local filesystem."""
        local_path = self.local_dir / filename
        local_path.write_text(data)

        location = str(local_path)
        logging.info(f"Saved locally: {location}")
        return location

    def read(self, filename: str) -> str:
        """
        Read data from configured storage.

        Args:
            filename: Filename to read

        Returns:
            File contents as string
        """
        if self.use_gcs:
            try:
                return self._read_from_gcs(filename)
            except Exception as e:
                logging.warning(f"GCS read failed, trying local: {e}")
                return self._read_from_local(filename)
        else:
            return self._read_from_local(filename)

    def _read_from_gcs(self, filename: str) -> str:
        """Read data from GCS."""
        from google.cloud import storage

        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.gcs_bucket)
        blob = bucket.blob(f'reports/{filename}')

        return blob.download_as_text()

    def _read_from_local(self, filename: str) -> str:
        """Read data from local filesystem."""
        local_path = self.local_dir / filename
        return local_path.read_text()

    def get_storage_info(self) -> str:
        """Get human-readable storage information."""
        if self.use_gcs:
            return f"GCS: gs://{self.gcs_bucket}/data/reports/"
        else:
            return f"Local: {self.local_dir}/"

    def upload_string_to_gcs(self, content: str, blob_name: str) -> None:
        """
        Upload string content to GCS.

        Args:
            content: String content to upload
            blob_name: Blob name/path in GCS (e.g., 'data/schemas/schema.json')
        """
        from google.cloud import storage

        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.gcs_bucket)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content, content_type='application/json')

        logging.info(f"Uploaded to GCS: gs://{self.gcs_bucket}/{blob_name}")

    def download_string_from_gcs(self, blob_name: str) -> str:
        """
        Download string content from GCS.

        Args:
            blob_name: Blob name/path in GCS

        Returns:
            String content
        """
        from google.cloud import storage

        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.gcs_bucket)
        blob = bucket.blob(blob_name)

        return blob.download_as_text()
