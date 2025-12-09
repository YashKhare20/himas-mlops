"""
Network Statistics Tools for Federated Coordinator

Provides aggregated, privacy-preserved statistics about the federated network.
"""

import logging
from datetime import datetime
from typing import Any

from .config import (
    PROJECT_ID,
    VERIFICATION_DATASET,
    HOSPITALS,
    get_table_path,
    ensure_serializable,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _safe_float(value, default=0.0):
    """Safely convert a value to Python float, handling numpy and None."""
    if value is None:
        return default
    try:
        import numpy as np
        if isinstance(value, (np.floating, np.integer)):
            if np.isnan(value):
                return default
            return float(value)
    except (ImportError, TypeError):
        pass
    try:
        result = float(value)
        if result != result:  # NaN check
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    """Safely convert a value to Python int, handling numpy and None."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_network_statistics() -> dict[str, Any]:
    """
    Get aggregated statistics from the entire federated network.

    Returns privacy-preserved statistics about the federated learning network.

    Returns:
        dict containing network-wide aggregated statistics
    """
    from google.cloud import bigquery

    try:
        client = bigquery.Client(project=PROJECT_ID)

        table_path = get_table_path(VERIFICATION_DATASET, "dataset_statistics")

        query = f"""
        SELECT 
            hospital,
            data_split,
            SUM(n_patients) as total_patients,
            SUM(n_deaths) as total_deaths,
            AVG(mortality_rate) as avg_mortality_rate,
            AVG(mean_age) as avg_age,
            AVG(mean_icu_los_hours) as avg_icu_los
        FROM `{table_path}`
        GROUP BY hospital, data_split
        ORDER BY hospital, data_split
        """

        result = client.query(query).to_dataframe()

        # Aggregate totals - ensure Python native types
        total_patients = _safe_int(result['total_patients'].sum(), 0)
        total_deaths = _safe_int(result['total_deaths'].sum(), 0)
        overall_mortality = float(
            total_deaths / total_patients) if total_patients > 0 else 0.0

        # Hospital summaries - convert numpy array to Python list
        hospital_summaries = {}
        # Convert numpy array to Python list
        unique_hospitals = [str(h)
                            for h in result['hospital'].unique().tolist()]

        for hospital in unique_hospitals:
            hospital_str = str(hospital)
            hospital_data = result[result['hospital'] == hospital]
            hospital_summaries[hospital_str] = {
                "total_patients": _safe_int(hospital_data['total_patients'].sum(), 0),
                "mortality_rate": float(round(_safe_float(hospital_data['avg_mortality_rate'].mean(), 0.0), 4)),
                "avg_age": float(round(_safe_float(hospital_data['avg_age'].mean(), 0.0), 1)),
                "avg_icu_los_hours": float(round(_safe_float(hospital_data['avg_icu_los'].mean(), 0.0), 1))
            }

        # Data split breakdown - ensure all values are Python native types
        split_breakdown = {}
        for split in ['train', 'validation', 'test']:
            split_data = result[result['data_split'] == split]
            if not split_data.empty:
                split_breakdown[str(split)] = {
                    "patients": _safe_int(split_data['total_patients'].sum(), 0),
                    "mortality_rate": float(round(_safe_float(split_data['avg_mortality_rate'].mean(), 0.0), 4))
                }

        response = {
            "network_summary": {
                "total_hospitals": int(len(HOSPITALS)),
                "hospital_ids": list(HOSPITALS),  # Make a copy
                "hospital_names": [
                    "Hospital A (Community)",
                    "Hospital B (Tertiary)",
                    "Hospital C (Rural)"
                ],
                "total_patients": int(total_patients),
                "total_icu_stays": int(total_patients),
                "overall_mortality_rate": float(round(overall_mortality, 4)),
                "last_training_date": "2025-01-15"
            },
            "hospital_summaries": hospital_summaries,
            "data_splits": split_breakdown,
            "privacy_note": "All statistics are aggregated and privacy-preserved",
            "dataset": str(VERIFICATION_DATASET),
            "timestamp": datetime.now().isoformat()
        }

        # Final safety check - ensure everything is serializable
        return ensure_serializable(response)

    except Exception as e:
        logger.error(f"Error getting network statistics: {str(e)}")
        return ensure_serializable({
            "error": str(e),
            "network_summary": None,
            "timestamp": datetime.now().isoformat()
        })
