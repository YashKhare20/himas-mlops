"""
Feature Extraction Utilities 

This file contains only helper functions for calculating timing flags.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Helper functions for feature calculation"""

    @staticmethod
    def calculate_early_icu_score(hours_admit_to_icu: float) -> int:
        """
        Calculate early ICU score based on hours from admission to ICU.

        Score mapping (matches clinical_features.sql):
        - < 6 hours  → 3 (Very early - most severe)
        - < 24 hours → 2 (Early)
        - < 48 hours → 1 (Delayed)
        - >= 48 hours → 0 (Late)
        """
        if hours_admit_to_icu < 6:
            return 3
        elif hours_admit_to_icu < 24:
            return 2
        elif hours_admit_to_icu < 48:
            return 1
        else:
            return 0

    @staticmethod
    def is_weekend(date_str: str = None) -> int:
        """Determines if weekend"""
        if not date_str:
            return 1 if datetime.now().weekday() >= 5 else 0

        # Handle semantic strings
        upper_str = str(date_str).upper()
        if any(x in upper_str for x in ['SATURDAY', 'SUNDAY', 'WEEKEND']):
            return 1
        if any(x in upper_str for x in ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'WEEKDAY']):
            return 0

        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            # Python: Monday=0, ..., Sunday=6
            # Weekend: Saturday(5) or Sunday(6)
            return 1 if dt.weekday() >= 5 else 0

        except Exception:
            logger.debug(
                f"Could not parse date '{date_str}', defaulting to current time check")
            return 1 if datetime.now().weekday() >= 5 else 0

    @staticmethod
    def is_night_admission(time_str: str = None) -> int:
        """Determines if night admission"""
        if not time_str:
            hour = datetime.now().hour
            return 1 if (hour >= 18 or hour < 6) else 0

        # Handle semantic strings
        upper_str = str(time_str).upper()
        if any(x in upper_str for x in ['NIGHT', 'EVENING', 'PM']):
            return 1
        if any(x in upper_str for x in ['MORNING', 'AFTERNOON', 'DAY', 'AM']):
            return 0

        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            hour = dt.hour
            return 1 if (hour >= 18 or hour < 6) else 0

        except Exception:
            logger.debug(
                f"Could not parse time '{time_str}', defaulting to current time check")
            hour = datetime.now().hour
            return 1 if (hour >= 18 or hour < 6) else 0
