"""
Test data leakage prevention mechanisms and schema validation setup

NOTE: Actual data leakage validation happens at runtime in the Airflow DAG
via DataValidator class. Schema validation ensures data quality over time.
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

    def test_dag_has_schema_validation_tasks(self):
        """Test that DAG includes schema validation task group"""
        from airflow.models import DagBag

        dag_bag = DagBag(dag_folder=DAG_FOLDER, include_examples=False)

        # Get DAG directly from dag_bag.dags (no database query)
        dag = dag_bag.dags.get('himas_bigquery_demo')

        if dag:
            task_ids = [task.task_id for task in dag.tasks]

            # Check for schema validation tasks
            has_schema_validation = any(
                'schema' in tid.lower() or
                'statistics' in tid.lower() or
                'drift' in tid.lower() or
                'quality' in tid.lower()
                for tid in task_ids
            )

            assert has_schema_validation, \
                "DAG should have schema validation tasks"

    def test_patient_split_configuration(self):
        """Test that patient split configuration is defined"""
        try:
            from utils.config import PipelineConfig
            config = PipelineConfig()

            # Verify config loads successfully
            assert config is not None, "PipelineConfig should be available"
            assert hasattr(
                config, 'PROJECT_ID'), "Config should have PROJECT_ID"

        except ImportError:
            pytest.skip("Config module not structured as expected")

    def test_hospital_count_configuration(self):
        """Test that expected number of hospitals is configured"""
        try:
            from utils.config import PipelineConfig
            config = PipelineConfig()

            # HIMAS expects 3 hospitals for federated learning
            if hasattr(config, 'FEDERATED_TABLES'):
                hospital_tables = [
                    t for t in config.FEDERATED_TABLES if 'hospital' in t]
                assert len(hospital_tables) == 3, \
                    "HIMAS should be configured for 3 hospitals"

        except ImportError:
            pytest.skip("Config module not structured as expected")

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


class TestSchemaValidationSetup:
    """Test that schema validation mechanisms are properly configured"""

    def test_schema_validator_module_exists(self):
        """Test that schema validator module exists"""
        schema_validator_file = Path(
            DAG_FOLDER) / 'utils' / 'schema_validator.py'
        assert schema_validator_file.exists(), \
            "schema_validator.py module should exist in dags/utils/"

    def test_schema_validator_class_exists(self):
        """Test that SchemaValidator class is available"""
        from utils.schema_validator import SchemaValidator
        assert SchemaValidator is not None, \
            "SchemaValidator class should be importable"

    def test_schema_validator_can_be_instantiated(self):
        """Test that SchemaValidator can be instantiated"""
        from utils.schema_validator import SchemaValidator

        # Skip if no GCP credentials (CI/CD environment)
        try:
            validator = SchemaValidator(
                project_id='test-project',
                location='US'
            )
            assert validator is not None
            assert validator.project_id == 'test-project'
            assert validator.location == 'US'
        except Exception as e:
            if 'DefaultCredentialsError' in str(type(e)):
                pytest.skip(
                    "No GCP credentials - skipping (expected in CI/CD)")
            else:
                raise

    def test_schema_validator_methods_exist(self):
        """Test that SchemaValidator has required methods"""
        from utils.schema_validator import SchemaValidator
        import inspect

        # Check methods exist without instantiating (avoids credentials requirement)
        methods = inspect.getmembers(
            SchemaValidator, predicate=inspect.isfunction)
        method_names = [name for name, _ in methods]

        assert 'extract_table_schema' in method_names, \
            "SchemaValidator should have extract_table_schema method"
        assert 'compute_table_statistics' in method_names, \
            "SchemaValidator should have compute_table_statistics method"
        assert 'detect_schema_drift' in method_names, \
            "SchemaValidator should have detect_schema_drift method"
        assert 'validate_data_quality' in method_names, \
            "SchemaValidator should have validate_data_quality method"

    def test_schema_utils_module_exists(self):
        """Test that schema utils helper module exists"""
        schema_utils_file = Path(DAG_FOLDER) / 'utils' / 'schema_utils.py'
        assert schema_utils_file.exists(), \
            "schema_utils.py module should exist in dags/utils/"

    def test_schema_utils_functions_exist(self):
        """Test that schema utils has required helper functions"""
        from utils.schema_utils import (
            extract_all_layer_schemas,
            compute_all_layer_statistics,
            detect_schema_drift_all_layers,
            validate_data_quality_all_layers,
            generate_comprehensive_quality_summary
        )

        assert extract_all_layer_schemas is not None
        assert compute_all_layer_statistics is not None
        assert detect_schema_drift_all_layers is not None
        assert validate_data_quality_all_layers is not None
        assert generate_comprehensive_quality_summary is not None

    def test_task_functions_module_has_schema_tasks(self):
        """Test that task functions module includes schema validation tasks"""
        from utils.task_functions import (
            create_extract_schemas_task_function,
            create_compute_statistics_task_function,
            create_detect_drift_task_function,
            create_validate_quality_task_function,
            create_quality_summary_task_function
        )

        assert create_extract_schemas_task_function is not None
        assert create_compute_statistics_task_function is not None
        assert create_detect_drift_task_function is not None
        assert create_validate_quality_task_function is not None
        assert create_quality_summary_task_function is not None

    def test_storage_handler_supports_gcs(self):
        """Test that StorageHandler supports GCS operations"""
        from utils.storage import StorageHandler

        # Test with local storage
        handler = StorageHandler(
            use_gcs=False,
            local_dir=Path('/tmp/test')
        )

        assert handler is not None
        assert handler.use_gcs == False

        # Check for required methods
        assert hasattr(handler, 'upload_string_to_gcs'), \
            "StorageHandler should have upload_string_to_gcs method"
        assert hasattr(handler, 'download_string_from_gcs'), \
            "StorageHandler should have download_string_from_gcs method"

    def test_config_has_storage_settings(self):
        """Test that config has storage configuration"""
        from utils.config import PipelineConfig

        config = PipelineConfig()

        assert hasattr(config, 'USE_GCS'), \
            "Config should have USE_GCS flag"
        assert hasattr(config, 'GCS_BUCKET'), \
            "Config should have GCS_BUCKET setting"


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
                'schema' in doc or
                'quality' in doc or
                'statistics' in doc
            )

            assert has_validation_docs, \
                "DAG documentation should mention data validation"

    def test_validator_classes_have_docstrings(self):
        """Test that validator classes have docstrings"""
        from utils.validation import DataValidator
        from utils.schema_validator import SchemaValidator
        import inspect

        # Check DataValidator (can instantiate without credentials)
        data_validator = DataValidator(project_id='test-project')
        assert data_validator.verify_data_integrity.__doc__ is not None, \
            "verify_data_integrity should have docstring"

        # Check SchemaValidator methods without instantiating
        methods = inspect.getmembers(
            SchemaValidator, predicate=inspect.isfunction)

        # Find extract_table_schema method
        extract_method = None
        for name, method in methods:
            if name == 'extract_table_schema':
                extract_method = method
                break

        assert extract_method is not None, \
            "extract_table_schema method should exist"
        assert extract_method.__doc__ is not None, \
            "extract_table_schema should have docstring"


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

    def test_layers_configuration(self):
        """Test that all required layers are configured"""
        from utils.config import PipelineConfig

        config = PipelineConfig()

        if hasattr(config, 'LAYERS'):
            # Should have curated, federated, and verification layers
            assert 'curated' in config.LAYERS, "Should have curated layer"
            assert 'federated' in config.LAYERS, "Should have federated layer"
            assert 'verification' in config.LAYERS, "Should have verification layer"


class TestDataQualityThresholds:
    """Test data quality validation thresholds"""

    def test_quality_thresholds_defined(self):
        """Test that quality thresholds are defined in task functions"""
        from utils.task_functions import validate_data_quality_task
        import inspect

        # Check that custom thresholds are defined
        source = inspect.getsource(validate_data_quality_task)

        assert 'row_count_change_pct' in source, \
            "Should define row count change threshold"
        assert 'null_rate_threshold' in source, \
            "Should define null rate threshold"
        assert 'distinct_ratio_min' in source, \
            "Should define distinct ratio threshold"

    def test_thresholds_are_reasonable(self):
        """Test that threshold values are reasonable for healthcare data"""
        from utils.task_functions import validate_data_quality_task
        import inspect

        source = inspect.getsource(validate_data_quality_task)

        # Extract threshold values (simple check)
        # Thresholds should be between 0 and 1 or reasonable percentages
        assert '50.0' in source or '0.5' in source, \
            "Row count change threshold should be reasonable (e.g., 50%)"
        assert '0.3' in source or '30' in source, \
            "Null rate threshold should be reasonable (e.g., 30%)"


class TestPipelineOutputs:
    """Test that pipeline generates expected outputs"""

    def test_output_directories_configured(self):
        """Test that output directories are properly configured"""
        expected_dirs = [
            'data/schemas',
            'data/statistics',
            'data/drift',
            'data/validation',
            'data/reports'
        ]

        # These directories should be created by the pipeline
        # Just verify the paths are consistent in code
        from utils.schema_utils import (
            extract_all_layer_schemas,
            compute_all_layer_statistics
        )
        import inspect

        extract_source = inspect.getsource(extract_all_layer_schemas)
        compute_source = inspect.getsource(compute_all_layer_statistics)

        assert 'data/schemas' in extract_source, \
            "Should use data/schemas directory"
        assert 'data/statistics' in compute_source, \
            "Should use data/statistics directory"

    def test_dvc_integration_configured(self):
        """Test that DVC integration is configured"""
        from utils.task_functions import (
            create_dvc_version_reports_task_function,
            create_dvc_version_all_data_task_function,
            create_dvc_version_bigquery_task_function
        )

        assert create_dvc_version_reports_task_function is not None
        assert create_dvc_version_all_data_task_function is not None
        assert create_dvc_version_bigquery_task_function is not None
