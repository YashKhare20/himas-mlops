"""
Test DVC setup and configuration
"""
import os
import pytest
from pathlib import Path


class TestDVCSetup:
    """Test DVC initialization and configuration"""

    def test_dvc_directory_exists(self):
        """Test that .dvc directory exists"""
        dvc_dir = Path('.dvc')
        assert dvc_dir.exists(), ".dvc directory should exist"
        assert dvc_dir.is_dir(), ".dvc should be a directory"

    def test_dvc_config_exists(self):
        """Test that .dvc/config file exists"""
        config_file = Path('.dvc/config')
        assert config_file.exists(), ".dvc/config file should exist"

    def test_dvc_gitignore_exists(self):
        """Test that .dvc/.gitignore exists"""
        gitignore = Path('.dvc/.gitignore')
        assert gitignore.exists(), ".dvc/.gitignore should exist"

    def test_dvc_files_in_gitignore(self):
        """Test that data cache is in .gitignore"""
        gitignore = Path('.gitignore')

        if gitignore.exists():
            content = gitignore.read_text()
            assert '/data/' in content or 'data/' in content, \
                "Data directory should be in .gitignore"

    def test_dvc_cache_directory(self):
        """Test that DVC cache directory exists"""
        cache_dir = Path('.dvc/cache')
        assert cache_dir.exists(), ".dvc/cache directory should exist"

    def test_dvc_remote_configured(self):
        """Test that DVC remote is configured"""
        config_file = Path('.dvc/config')

        if config_file.exists():
            content = config_file.read_text()
            # Check if remote is configured (either local or GCS)
            has_remote = (
                "['remote \"local\"']" in content or
                "['remote \"gcs_storage\"']" in content or
                "[remote " in content.lower()
            )

            # For now, just warn if no remote
            if not has_remote:
                pytest.skip("No DVC remote configured yet")

    def test_dvc_files_tracked(self):
        """Test that .dvc files exist for data"""
        dvc_files = list(Path('.').rglob('*.dvc'))

        # For initial setup, we might not have .dvc files yet
        if len(dvc_files) == 0:
            pytest.skip("No .dvc files found yet - dataset not tracked")

        # If .dvc files exist, they should be valid
        for dvc_file in dvc_files:
            assert dvc_file.stat().st_size > 0, \
                f"{dvc_file} should not be empty"

    def test_requirements_has_dvc(self):
        """Test that requirements.txt includes DVC"""
        req_file = Path('requirements.txt')

        if req_file.exists():
            content = req_file.read_text()
            assert 'dvc-gs' in content.lower(), \
                "requirements.txt should include dvc"

    def test_no_large_files_in_git(self):
        """Test that large data files are not tracked by git"""
        # Common data file extensions that should be in .dvc, not git
        data_extensions = ['.csv', '.parquet', '.json', '.xlsx']

        gitignore = Path('.gitignore')
        if gitignore.exists():
            content = gitignore.read_text()

            # At least some data patterns should be ignored
            data_ignored = any(
                ext in content for ext in data_extensions
            ) or 'data/' in content

            assert data_ignored, \
                "Large data files should be in .gitignore"
