"""
SQL utilities for HIMAS Pipeline.

Handles SQL file loading and processing.
"""
import logging
from pathlib import Path
from typing import List, Dict


class SQLFileLoader:
    """Handles loading and processing SQL files."""

    def __init__(self, sql_dir: Path):
        """
        Initialize SQL file loader.

        Args:
            sql_dir: Path to SQL directory containing layer subdirectories
        """
        self.sql_dir = Path(sql_dir)

        if not self.sql_dir.exists():
            raise ValueError(f"SQL directory not found: {self.sql_dir}")

    def get_layer_files(self, layer_name: str) -> List[Path]:
        """
        Get sorted list of SQL files from a layer directory.

        Args:
            layer_name: Name of the SQL layer directory (e.g., 'curated_layer')

        Returns:
            List of Path objects for SQL files, sorted by filename
        """
        layer_path = self.sql_dir / layer_name

        if not layer_path.exists():
            logging.warning(f"SQL layer directory not found: {layer_path}")
            return []

        sql_files = sorted(layer_path.glob('*.sql'))
        logging.info(f"Found {len(sql_files)} SQL files in {layer_name}")

        return sql_files

    def read_sql(self, sql_file: Path) -> str:
        """
        Read SQL file contents.

        Args:
            sql_file: Path to SQL file

        Returns:
            SQL query as string
        """
        if not sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_file}")

        return sql_file.read_text()

    def get_layer_queries(self, layer_name: str) -> List[Dict]:
        """
        Get all SQL queries from a layer with metadata.

        Args:
            layer_name: Name of the SQL layer directory

        Returns:
            List of dictionaries with 'name', 'path', and 'sql' keys
        """
        sql_files = self.get_layer_files(layer_name)

        queries = []
        for sql_file in sql_files:
            queries.append({
                'name': sql_file.stem,
                'path': sql_file,
                'sql': self.read_sql(sql_file)
            })

        return queries

    @staticmethod
    def validate_sql_basic(sql: str) -> bool:
        """
        Basic SQL validation (checks for common issues).

        Args:
            sql: SQL query string

        Returns:
            True if basic validation passes
        """
        if not sql or not sql.strip():
            return False

        sql_upper = sql.upper()

        # Check for dangerous operations in production
        dangerous_keywords = ['DROP TABLE', 'DROP DATASET', 'DELETE FROM']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                logging.warning(f"Potentially dangerous operation: {keyword}")

        return True
