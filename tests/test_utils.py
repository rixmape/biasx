"""Tests for the utility functions in the BiasX library."""

import json
import os
import pathlib
import tempfile
from unittest.mock import patch, mock_open

import pytest
from biasx.utils import (
    get_json_config,
    get_resource_path,
    get_cache_dir,
    get_file_path
)


class TestGetJsonConfig:
    def test_get_json_config_success(self, tmp_path):
        """Test that get_json_config loads JSON correctly."""
        # Create a temporary JSON file
        # Create a fake module file to serve as the caller_file
        module_file = tmp_path / "module.py"
        module_file.touch()
        
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "test_config.json"
        test_data = {"key": "value", "number": 42}
        
        with open(test_file, "w") as f:
            json.dump(test_data, f)
        
        # Test loading the config - pass the module file path instead of directory
        result = get_json_config(str(module_file), "test_config.json")
        assert result == test_data
        
    def test_get_json_config_file_not_found(self):
        """Test that get_json_config raises FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            get_json_config("/non/existent/path", "not_existing.json")
            
    def test_get_json_config_caching(self, tmp_path):
        """Test that get_json_config uses cache for repeated calls."""
        # Create a fake module file to serve as the caller_file
        module_file = tmp_path / "module.py"
        module_file.touch()
        
        # Create a temporary JSON file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        test_file = data_dir / "test_config.json"
        test_data = {"key": "value"}
        
        with open(test_file, "w") as f:
            json.dump(test_data, f)
        
        # Call the function twice with the same arguments
        result1 = get_json_config(str(module_file), "test_config.json")
        
        # Change the file content
        with open(test_file, "w") as f:
            json.dump({"key": "new_value"}, f)
            
        # Second call should return cached result, not new content
        result2 = get_json_config(str(module_file), "test_config.json")
        
        assert result1 == result2
        assert result1["key"] == "value"  # Not "new_value"


class TestGetCacheDir:
    def test_get_cache_dir_creates_directory(self):
        """Test that get_cache_dir creates the directory if it doesn't exist."""
        # Use a unique temporary name
        test_dir_name = f"test_cache_{os.getpid()}"
        
        # Get the expected path
        home = pathlib.Path(os.path.expanduser("~"))
        expected_path = home / ".biasx" / "cache" / test_dir_name
        
        # Ensure the directory doesn't exist initially
        if expected_path.exists():
            import shutil
            shutil.rmtree(expected_path)
            
        try:
            # Call the function
            result = get_cache_dir(test_dir_name)
            
            # Check the result
            assert result == expected_path
            assert expected_path.exists()
            assert expected_path.is_dir()
            
        finally:
            # Clean up
            if expected_path.exists():
                import shutil
                shutil.rmtree(expected_path)
                
    def test_get_cache_dir_existing_directory(self):
        """Test that get_cache_dir works with an existing directory."""
        # Use a unique temporary name
        test_dir_name = f"test_existing_{os.getpid()}"
        
        # Get the expected path and create it manually
        home = pathlib.Path(os.path.expanduser("~"))
        expected_path = home / ".biasx" / "cache" / test_dir_name
        expected_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Call the function
            result = get_cache_dir(test_dir_name)
            
            # Check the result
            assert result == expected_path
            
        finally:
            # Clean up
            if expected_path.exists():
                import shutil
                shutil.rmtree(expected_path)


class TestGetFilePath:
    def test_get_file_path_existing_file(self, tmp_path):
        """Test that get_file_path returns the correct path for an existing file."""
        # Create a fake module file to serve as the caller_file
        module_file = tmp_path / "module.py"
        module_file.touch()
        
        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Call the function with the module file path
        result = get_file_path(str(module_file), "test.txt")
        
        # Check the result
        assert result == test_file
        
    def test_get_file_path_file_not_found(self, tmp_path):
        """Test that get_file_path raises FileNotFoundError when file doesn't exist."""
        # Create a fake module file
        module_file = tmp_path / "module.py"
        module_file.touch()
        
        with pytest.raises(FileNotFoundError):
            get_file_path(str(module_file), "non_existent.txt")


@patch('biasx.utils.hf_hub_download')
class TestGetResourcePath:
    def test_get_resource_path_calls_hf_hub_download(self, mock_hf_download):
        """Test that get_resource_path calls hf_hub_download with correct parameters."""
        # Setup the mock
        mock_hf_download.return_value = "/path/to/downloaded/file"
        
        # Call the function
        result = get_resource_path("test-repo", "test.txt")
        
        # Check the result
        assert result == "/path/to/downloaded/file"
        
        # Verify the mock was called correctly
        mock_hf_download.assert_called_once()
        call_args = mock_hf_download.call_args[1]
        assert call_args["repo_id"] == "test-repo"
        assert call_args["filename"] == "test.txt"
        assert call_args["repo_type"] == "dataset"
        assert not call_args["force_download"]
        
    def test_get_resource_path_with_custom_parameters(self, mock_hf_download):
        """Test that get_resource_path passes custom parameters to hf_hub_download."""
        # Setup the mock
        mock_hf_download.return_value = "/path/to/downloaded/file"
        
        # Call the function with custom parameters
        result = get_resource_path(
            "test-repo",
            "test.txt",
            repo_type="model",
            force_download=True
        )
        
        # Check the result
        assert result == "/path/to/downloaded/file"
        
        # Verify the mock was called correctly
        call_args = mock_hf_download.call_args[1]
        assert call_args["repo_type"] == "model"
        assert call_args["force_download"] is True