"""
BigQuery Client for Hospital A
"""

import logging
from typing import Dict, Any, List
from google.cloud import bigquery

logger = logging.getLogger(__name__)


class HospitalBigQueryClient:
    """
    BigQuery client with hospital-specific access restrictions.
    """
    
    def __init__(self, project_id: str, dataset_id: str, table_id: str):
        """Initialize BigQuery client."""
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = bigquery.Client(project=project_id)
        self.full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        
        logger.info(f"BigQuery client: {self.full_table_id}")
    
    def query(self, query: str) -> List[Dict[str, Any]]:
        """Executes BigQuery query."""
        try:
            query_job = self.client.query(query)
            results = list(query_job.result())
            logger.info(f"Query returned {len(results)} rows")
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise