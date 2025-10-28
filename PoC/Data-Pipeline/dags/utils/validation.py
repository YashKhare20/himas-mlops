"""
Data validation utilities for HIMAS Pipeline.

Handles data integrity checks and quality validation.
"""
import logging
from typing import Dict, Any
import pandas as pd


class DataValidator:
    """Handles data validation operations."""

    def __init__(self, project_id: str, location: str = 'US'):
        """
        Initialize data validator.

        Args:
            project_id: GCP project ID
            location: BigQuery location
        """
        self.project_id = project_id
        self.location = location

    def verify_data_integrity(self, storage_handler, **context) -> Dict[str, Any]:
        """
        Verify data integrity - no patient overlap between splits.

        Args:
            storage_handler: StorageHandler instance for saving reports
            **context: Airflow context dictionary

        Returns:
            Dictionary with validation results

        Raises:
            ValueError: If data leakage is detected
        """
        from google.cloud import bigquery

        logging.info("Verifying data integrity...")

        client = bigquery.Client(
            project=self.project_id, location=self.location)

        # Query leakage check view
        query = f"""
        SELECT *
        FROM `{self.project_id}.verification_demo.data_leakage_check`
        ORDER BY hospital
        """

        df_leakage = client.query(query).to_dataframe()

        # Check for overlaps
        overlap_columns = ['train_val_overlap',
                           'train_test_overlap', 'val_test_overlap']
        total_overlap = df_leakage[overlap_columns].sum().sum()

        if total_overlap > 0:
            error_msg = (
                f"DATA LEAKAGE DETECTED: {total_overlap} overlapping patients!\n"
                f"{df_leakage.to_string()}"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Save reports
        csv_location = storage_handler.save(
            df_leakage.to_csv(index=False),
            'data_leakage_check.csv',
            'text/csv'
        )

        json_location = storage_handler.save(
            df_leakage.to_json(orient='records', indent=2),
            'data_leakage_check.json',
            'application/json'
        )

        logging.info("Zero data leakage verified")

        # Return results
        results = {
            'passed': True,
            'total_overlap': int(total_overlap),
            'csv_location': csv_location,
            'json_location': json_location
        }

        # Store in XCom
        if context:
            context['task_instance'].xcom_push(
                key='leakage_check_passed', value=True)
            context['task_instance'].xcom_push(
                key='report_location', value=csv_location)

        return results

    def generate_statistics(self, storage_handler, **context) -> Dict[str, Any]:
        """
        Generate comprehensive dataset statistics.

        Args:
            storage_handler: StorageHandler instance for saving reports
            **context: Airflow context dictionary

        Returns:
            Dictionary with statistics and file locations
        """
        from google.cloud import bigquery

        logging.info("Generating dataset statistics...")

        client = bigquery.Client(
            project=self.project_id, location=self.location)

        # Query statistics view
        query = f"""
        SELECT *
        FROM `{self.project_id}.verification_demo.dataset_statistics`
        ORDER BY hospital, data_split
        """

        df_stats = client.query(query).to_dataframe()

        # Save reports
        csv_location = storage_handler.save(
            df_stats.to_csv(index=False),
            'dataset_statistics.csv',
            'text/csv'
        )

        json_location = storage_handler.save(
            df_stats.to_json(orient='records', indent=2),
            'dataset_statistics.json',
            'application/json'
        )

        # Log key metrics
        logging.info("Dataset Statistics:")
        for _, row in df_stats.iterrows():
            if row['data_split'] == 'train':
                logging.info(
                    f"   {row['hospital']}: {row['n_patients']} patients, "
                    f"mortality rate = {row['mortality_rate']:.1%}"
                )

        results = {
            'csv_location': csv_location,
            'json_location': json_location,
            'total_records': len(df_stats)
        }

        # Store in XCom
        if context:
            context['task_instance'].xcom_push(
                key='statistics_location', value=csv_location)

        return results
