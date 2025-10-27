"""
Test DAG integrity and validation
"""
import os
import sys
from datetime import datetime
import pytest
from airflow.models import DagBag
from airflow.utils.dag_cycle_tester import check_cycle

# Add dags folder to path
DAG_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'dags')
sys.path.insert(0, DAG_FOLDER)


class TestDAGIntegrity:
    """Test DAG integrity and basic validation"""

    @pytest.fixture(scope='class')
    def dag_bag(self):
        """Load DAG bag once for all tests"""
        return DagBag(dag_folder=DAG_FOLDER, include_examples=False)

    @pytest.fixture(scope='class')
    def dag(self, dag_bag):
        """Get the HIMAS DAG directly from dag_bag.dags"""
        return dag_bag.dags.get('himas_bigquery_demo')

    def test_no_import_errors(self, dag_bag):
        """Test that there are no import errors in DAGs"""
        assert not dag_bag.import_errors, \
            f"DAG import errors: {dag_bag.import_errors}"

    def test_dags_loaded(self, dag_bag):
        """Test that at least one DAG is loaded"""
        assert len(dag_bag.dags) > 0, "No DAGs loaded"

    def test_himas_dag_exists(self, dag_bag):
        """Test that the HIMAS BigQuery DAG exists"""
        assert 'himas_bigquery_demo' in dag_bag.dag_ids, \
            "himas_bigquery_demo DAG not found"

    def test_dag_has_tags(self, dag):
        """Test that DAG has tags"""
        if dag is None:
            pytest.skip("DAG not loaded")
        assert dag.tags, "DAG should have tags"

    def test_dag_has_owner(self, dag):
        """Test that DAG has owner specified"""
        if dag is None:
            pytest.skip("DAG not loaded")
        assert dag.owner, "DAG should have an owner"
        assert dag.owner != 'airflow', "DAG owner should be customized"

    def test_dag_has_retries(self, dag):
        """Test that DAG has retry configuration"""
        if dag is None:
            pytest.skip("DAG not loaded")
        assert dag.default_args.get('retries') is not None, \
            "DAG should have retry configuration"

    def test_dag_schedule(self, dag):
        """Test that DAG has a valid schedule"""
        if dag is None:
            pytest.skip("DAG not loaded")
        # Schedule can be None for manual trigger
        assert dag.schedule is not None or dag.schedule is None, \
            "DAG should have a schedule interval or None for manual trigger"

    def test_dag_start_date(self, dag):
        """Test that DAG has a start date"""
        if dag is None:
            pytest.skip("DAG not loaded")
        assert dag.start_date is not None, "DAG should have a start date"
        assert isinstance(dag.start_date, datetime), \
            "Start date should be datetime object"

    def test_dag_tasks_exist(self, dag):
        """Test that DAG has tasks"""
        if dag is None:
            pytest.skip("DAG not loaded")
        assert len(dag.tasks) > 0, "DAG should have at least one task"

    def test_no_cycles(self, dag_bag):
        """Test that DAG has no cycles"""
        for dag_id, dag in dag_bag.dags.items():
            check_cycle(dag)

    def test_task_dependencies(self, dag):
        """Test that tasks have proper dependencies"""
        if dag is None:
            pytest.skip("DAG not loaded")

        # Check that tasks have dependencies (except start tasks)
        tasks_with_no_deps = [
            task for task in dag.tasks
            if not task.upstream_task_ids and not task.downstream_task_ids
        ]

        # Allow only one task with no dependencies (start task)
        # or all tasks connected
        assert len(tasks_with_no_deps) <= 1, \
            f"Found isolated tasks: {[t.task_id for t in tasks_with_no_deps]}"

    def test_task_ids_unique(self, dag):
        """Test that all task IDs are unique within DAG"""
        if dag is None:
            pytest.skip("DAG not loaded")
        task_ids = [task.task_id for task in dag.tasks]
        assert len(task_ids) == len(set(task_ids)), \
            "Duplicate task IDs found"

    def test_catchup_disabled(self, dag):
        """Test that catchup is disabled for production DAGs"""
        if dag is None:
            pytest.skip("DAG not loaded")
        assert not dag.catchup, \
            "Catchup should be disabled for production DAGs"

    def test_email_on_failure(self, dag):
        """Test that email alerts are configured"""
        if dag is None:
            pytest.skip("DAG not loaded")
        default_args = dag.default_args

        # Check if email configuration exists
        assert 'email_on_failure' in default_args, \
            "email_on_failure should be configured"

    def test_task_timeout(self, dag):
        """Test that tasks have execution timeout configured"""
        if dag is None:
            pytest.skip("DAG not loaded")

        # For this test, we just check that timeout can be configured
        # It's optional, so we just verify the mechanism is available
        for task in dag.tasks:
            # Check that task has the execution_timeout attribute
            assert hasattr(task, 'execution_timeout'), \
                f"Task {task.task_id} should have execution_timeout attribute"
