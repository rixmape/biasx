"""Tests for the utility functions in the BiasX library."""

import json
import os
import pathlib
import tempfile
from unittest.mock import patch

import pytest
from biasx.utils import (
    get_json_config,
    get_resource_path,
    get_cache_dir,
    get_file_path
)


class TestGetJsonConfig:
    def test_get_json_config_success(self, tmp_path):
        """Test that get_json_config correctly loads a JSON configuration file."""
        # Setup: Create a test module file and a JSON config file
        module_file = tmp_path / "module.py"
        module_file.touch()
        
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "test_config.json"
        test_data = {"key": "value", "number": 42}
        
        with open(test_file, "w") as f:
            json.dump(test_data, f)
        
        # Act: Load the config using the utility function
        result = get_json_config(str(module_file), "test_config.json")
        
        # Assert: The loaded data matches what was written
        assert result == test_data
        
    def test_get_json_config_file_not_found(self):
        """Test that get_json_config raises an error when the file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            get_json_config("/non/existent/path", "not_existing.json")
            
    def test_get_json_config_caching(self, tmp_path):
        """Test that get_json_config caches results for repeated calls."""
        # Setup: Create test module and JSON file
        module_file = tmp_path / "module.py"
        module_file.touch()
        
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "test_config.json"
        test_data = {"key": "value"}
        
        with open(test_file, "w") as f:
            json.dump(test_data, f)
        
        # First call should read from file
        result1 = get_json_config(str(module_file), "test_config.json")
        
        # Modify the file content
        with open(test_file, "w") as f:
            json.dump({"key": "new_value"}, f)
            
        # Second call should return cached result instead of reading again
        result2 = get_json_config(str(module_file), "test_config.json")
        
        # Assert: Both calls return the same object and the original value
        assert result1 == result2
        assert result1["key"] == "value"  # Not the updated "new_value"


class TestGetCacheDir:
    def test_get_cache_dir_creates_directory(self):
        """Test that get_cache_dir creates the directory if it doesn't exist."""
        # Setup: Generate a unique directory name
        test_dir_name = f"test_cache_{os.getpid()}"
        
        # Calculate expected path
        home = pathlib.Path(os.path.expanduser("~"))
        expected_path = home / ".biasx" / "cache" / test_dir_name
        
        # Ensure directory doesn't exist initially
        if expected_path.exists():
            import shutil
            shutil.rmtree(expected_path)
            
        try:
            # Act: Call the function
            result = get_cache_dir(test_dir_name)
            
            # Assert: Directory was created with correct path
            assert result == expected_path
            assert expected_path.exists()
            assert expected_path.is_dir()
            
        finally:
            # Cleanup: Remove test directory
            if expected_path.exists():
                import shutil
                shutil.rmtree(expected_path)
                
    def test_get_cache_dir_existing_directory(self):
        """Test that get_cache_dir works with an existing directory."""
        # Setup: Create unique directory manually
        test_dir_name = f"test_existing_{os.getpid()}"
        
        home = pathlib.Path(os.path.expanduser("~"))
        expected_path = home / ".biasx" / "cache" / test_dir_name
        expected_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Act: Call function on existing directory
            result = get_cache_dir(test_dir_name)
            
            # Assert: Returns correct path
            assert result == expected_path
            
        finally:
            # Cleanup
            if expected_path.exists():
                import shutil
                shutil.rmtree(expected_path)


class TestGetFilePath:
    def test_get_file_path_existing_file(self, tmp_path):
        """Test that get_file_path returns the correct path for an existing file."""
        # Setup: Create module file and test file
        module_file = tmp_path / "module.py"
        module_file.touch()
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Act: Get file path
        result = get_file_path(str(module_file), "test.txt")
        
        # Assert: Path is correctly resolved
        assert result == test_file
        
    def test_get_file_path_file_not_found(self, tmp_path):
        """Test that get_file_path raises an error when file doesn't exist."""
        # Setup: Create module file without the target file
        module_file = tmp_path / "module.py"
        module_file.touch()
        
        # Act & Assert: Function raises appropriate error
        with pytest.raises(FileNotFoundError):
            get_file_path(str(module_file), "non_existent.txt")


@patch('biasx.utils.hf_hub_download')
class TestGetResourcePath:
    def test_get_resource_path_calls_hf_hub_download(self, mock_hf_download):
        """Test that get_resource_path calls HuggingFace download with correct parameters."""
        # Setup: Configure mock to return a path
        mock_hf_download.return_value = "/path/to/downloaded/file"
        
        # Act: Call the function
        result = get_resource_path("test-repo", "test.txt")
        
        # Assert: Result is correct and function called with expected args
        assert result == "/path/to/downloaded/file"
        
        mock_hf_download.assert_called_once()
        call_args = mock_hf_download.call_args[1]
        assert call_args["repo_id"] == "test-repo"
        assert call_args["filename"] == "test.txt"
        assert call_args["repo_type"] == "dataset"
        assert not call_args["force_download"]
        
    def test_get_resource_path_with_custom_parameters(self, mock_hf_download):
        """Test that get_resource_path passes custom parameters to the download function."""
        # Setup: Configure mock
        mock_hf_download.return_value = "/path/to/downloaded/file"
        
        # Act: Call with custom parameters
        result = get_resource_path(
            "test-repo",
            "test.txt",
            repo_type="model",
            force_download=True
        )
        
        # Assert: Custom parameters were passed correctly
        assert result == "/path/to/downloaded/file"
        
        call_args = mock_hf_download.call_args[1]
        assert call_args["repo_type"] == "model"
        assert call_args["force_download"] is True