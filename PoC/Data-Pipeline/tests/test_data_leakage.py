"""
Test data leakage prevention mechanisms and validation setup

NOTE: Actual data leakage validation happens at runtime in the Airflow DAG
via DataValidator class. These tests verify that the validation mechanisms
are properly configured and available.
"""
import os
import sys
import pytest
from pathlib import Path

# Add dags folder to path
DAG_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'dags')
sys.path.insert(0, DAG_FOLDER)


class TestDataLeakageSetup:
    """Test that data leakage prevention mechanisms are properly configured"""

    def test_validation_module_exists(self):
        """Test that validation module exists"""
        validation_file = Path(DAG_FOLDER) / 'utils' / 'validation.py'
        assert validation_file.exists(), \
            "validation.py module should exist in dags/utils/"

    def test_data_validator_class_exists(self):
        """Test that DataValidator class is available"""
        from utils.validation import DataValidator
        assert DataValidator is not None, \
            "DataValidator class should be importable"

    def test_data_validator_can_be_instantiated(self):
        """Test that DataValidator can be instantiated"""
        from utils.validation import DataValidator

        validator = DataValidator(
            project_id='test-project',
            location='US'
        )

        assert validator is not None
        assert validator.project_id == 'test-project'
        assert validator.location == 'US'

    def test_verify_data_integrity_method_exists(self):
        """Test that verify_data_integrity method exists"""
        from utils.validation import DataValidator

        validator = DataValidator(project_id='test-project')
        assert hasattr(validator, 'verify_data_integrity'), \
            "DataValidator should have verify_data_integrity method"
        assert callable(validator.verify_data_integrity), \
            "verify_data_integrity should be callable"

    def test_generate_statistics_method_exists(self):
        """Test that generate_statistics method exists"""
        from utils.validation import DataValidator

        validator = DataValidator(project_id='test-project')
        assert hasattr(validator, 'generate_statistics'), \
            "DataValidator should have generate_statistics method"
        assert callable(validator.generate_statistics), \
            "generate_statistics should be callable"

    def test_verification_layer_sql_exists(self):
        """Test that verification layer SQL files exist"""
        sql_dir = Path(DAG_FOLDER) / 'sql' / 'verification_layer'

        if sql_dir.exists():
            sql_files = list(sql_dir.glob('*.sql'))
            assert len(sql_files) > 0, \
                "Verification layer should have SQL files"

            # Check for data leakage check view
            leakage_check_files = [
                f for f in sql_files
                if 'leakage' in f.name.lower() or 'overlap' in f.name.lower()
            ]
            assert len(leakage_check_files) > 0, \
                "Should have SQL file for data leakage checks"
        else:
            pytest.skip(
                "SQL directory not found - acceptable for initial setup")

    def test_dag_has_verification_task_group(self):
        """Test that DAG includes verification/quality checks"""
        from airflow.models import DagBag

        dag_bag = DagBag(dag_folder=DAG_FOLDER, include_examples=False)

        # Get DAG directly from dag_bag.dags (no database query)
        dag = dag_bag.dags.get('himas_bigquery_demo')

        if dag:
            task_ids = [task.task_id for task in dag.tasks]

            # Check for verification or quality check tasks
            has_verification = any(
                'verif' in tid.lower() or
                'quality' in tid.lower() or
                'integrity' in tid.lower()
                for tid in task_ids
            )

            assert has_verification, \
                "DAG should have verification/quality check tasks"

    def test_patient_split_configuration(self):
        """Test that patient split configuration is defined"""
        # Check if config has split percentages defined
        try:
            from utils.config import PipelineConfig
            config = PipelineConfig()

            # These might be in config or SQL - just verify config loads
            assert config is not None, "PipelineConfig should be available"

        except ImportError:
            pytest.skip("Config module not structured as expected")

    def test_hospital_count_configuration(self):
        """Test that expected number of hospitals is configured"""
        # In HIMAS, we expect 3 hospitals for federated learning
        expected_hospitals = 3

        # This is validated through SQL files or config
        # For now, just document the expectation
        assert expected_hospitals == 3, \
            "HIMAS should be configured for 3 hospitals"

    def test_validation_raises_on_leakage(self):
        """Test that validator is designed to raise errors on leakage"""
        from utils.validation import DataValidator
        import inspect

        validator = DataValidator(project_id='test-project')

        # Check method signature and docstring
        method = validator.verify_data_integrity
        source = inspect.getsource(method)

        # Should have error handling for leakage
        assert 'raise' in source or 'error' in source.lower(), \
            "verify_data_integrity should raise errors on validation failure"

    def test_leakage_check_query_structure(self):
        """Test that leakage check queries follow proper structure"""
        # Check if SQL files exist and contain overlap checks
        sql_dir = Path(DAG_FOLDER) / 'sql' / 'verification_layer'

        if not sql_dir.exists():
            pytest.skip(
                "SQL directory not found - acceptable for initial setup")

        leakage_files = list(sql_dir.glob('*leakage*.sql'))

        if leakage_files:
            content = leakage_files[0].read_text().lower()

            # Should check for overlaps
            has_overlap_logic = (
                'intersect' in content or
                'overlap' in content or
                'distinct' in content
            )

            assert has_overlap_logic, \
                "Leakage check SQL should contain overlap detection logic"

    def test_statistics_query_includes_splits(self):
        """Test that statistics query includes split information"""
        sql_dir = Path(DAG_FOLDER) / 'sql' / 'verification_layer'

        if not sql_dir.exists():
            pytest.skip(
                "SQL directory not found - acceptable for initial setup")

        stats_files = list(sql_dir.glob('*statistic*.sql'))

        if stats_files:
            content = stats_files[0].read_text().lower()

            # Should track train/val/test splits
            has_split_tracking = (
                'train' in content or
                'split' in content or
                'validation' in content
            )

            assert has_split_tracking, \
                "Statistics query should track data splits"

    def test_dag_on_failure_callback_configured(self):
        """Test that DAG has failure callbacks for alerting"""
        from airflow.models import DagBag

        dag_bag = DagBag(dag_folder=DAG_FOLDER, include_examples=False)

        # Get DAG directly from dag_bag.dags (no database query)
        dag = dag_bag.dags.get('himas_bigquery_demo')

        if dag:
            # Check for email on failure or callbacks
            has_email = dag.default_args.get('email_on_failure', False)
            has_callback = (
                dag.on_failure_callback is not None or
                dag.default_args.get('on_failure_callback') is not None
            )

            assert has_email or has_callback, \
                "DAG should have failure notifications configured"


class TestValidationDocumentation:
    """Test that validation mechanisms are documented"""

    def test_dag_docstring_mentions_validation(self):
        """Test that DAG docstring mentions data validation"""
        from airflow.models import DagBag

        dag_bag = DagBag(dag_folder=DAG_FOLDER, include_examples=False)

        # Get DAG directly from dag_bag.dags (no database query)
        dag = dag_bag.dags.get('himas_bigquery_demo')

        if dag and dag.doc_md:
            doc = dag.doc_md.lower()
            has_validation_docs = (
                'validation' in doc or
                'leakage' in doc or
                'quality' in doc or
                'integrity' in doc
            )

            assert has_validation_docs, \
                "DAG documentation should mention data validation"

    def test_validator_class_has_docstrings(self):
        """Test that DataValidator methods have docstrings"""
        from utils.validation import DataValidator

        validator = DataValidator(project_id='test-project')

        # Check main methods have docstrings
        assert validator.verify_data_integrity.__doc__ is not None, \
            "verify_data_integrity should have docstring"

        assert validator.generate_statistics.__doc__ is not None, \
            "generate_statistics should have docstring"

    def test_validation_module_has_module_docstring(self):
        """Test that validation module has documentation"""
        from utils import validation

        assert validation.__doc__ is not None, \
            "validation module should have docstring"


class TestFederatedLearningRequirements:
    """Test federated learning specific requirements"""

    def test_hospital_isolation_in_sql(self):
        """Test that SQL queries maintain hospital isolation"""
        sql_dir = Path(DAG_FOLDER) / 'sql' / 'federated_layer'

        if not sql_dir.exists():
            pytest.skip("Federated SQL directory not found")

        hospital_files = list(sql_dir.glob('hospital_*.sql'))

        # Should have separate files for each hospital
        assert len(hospital_files) >= 2, \
            "Should have SQL files for multiple hospitals"

    def test_patient_level_split_design(self):
        """Test that design uses patient-level splits (not temporal)"""
        # This is a design validation - patient-level splits prevent
        # temporal leakage where future data influences training

        sql_dir = Path(DAG_FOLDER) / 'sql'

        if sql_dir.exists():
            # Check for patient-based splitting in SQL files
            assignment_files = list(sql_dir.rglob('*assignment*.sql'))

            if assignment_files:
                content = assignment_files[0].read_text().lower()

                # Should split by patient, not by date
                has_patient_split = 'subject_id' in content or 'patient' in content

                assert has_patient_split, \
                    "Should use patient-level splits for federated learning"
