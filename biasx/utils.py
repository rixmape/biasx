"""Utility functions and common operations for the BiasX library."""

import json
import os
import pathlib
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, TypeVar

from huggingface_hub import hf_hub_download

T = TypeVar("T", bound=Enum)


@lru_cache(maxsize=64)
def get_json_config(caller_file: str, config_file: str) -> Dict[str, Any]:
    """Load a JSON configuration file relative to the calling module's data directory.

    Constructs the path to the config file assuming it resides within a 'data'
    subdirectory relative to the Python file that calls this function.
    Uses LRU caching to avoid reloading the same configuration file multiple times.

    Args:
        caller_file (str): The path to the Python file calling this function
                           (typically `__file__` from the caller). Used to determine
                           the module's directory.
        config_file (str): The name of the JSON configuration file (e.g.,
                           'dataset_config.json').

    Returns:
        A dictionary containing the parsed JSON configuration data.

    Raises:
        FileNotFoundError: If the constructed path to the configuration file
                           does not exist.
        json.JSONDecodeError: If the file exists but is not valid JSON.
    """
    module_dir = pathlib.Path(caller_file).parent
    data_path = module_dir / "data" / config_file

    if not data_path.exists():
        raise FileNotFoundError(f"File not found at {data_path}")

    with open(data_path, "r") as f:
        return json.load(f)


@lru_cache(maxsize=64)
def get_resource_path(repo_id: str, filename: str, repo_type: str = "dataset", force_download: bool = False) -> str:
    """Download or retrieve a cached resource file from HuggingFace Hub.

    Uses the `huggingface_hub` library to download a file from a specified
    repository. Manages caching in a standardized local directory structure
    within `~/.biasx/cache/`. Uses LRU caching on the function call itself
    to quickly return the path if requested again with the same arguments.

    Args:
        repo_id (str): The HuggingFace Hub repository ID (e.g., 'google/mediapipe').
        filename (str): The name of the file to download from the repository.
        repo_type (str, optional): The type of the repository (e.g., 'dataset',
                                   'model', 'space'). Defaults to "dataset".
        force_download (bool, optional): If True, forces re-downloading the file
                                         even if it exists in the cache. Defaults to False.

    Returns:
        The local file path to the downloaded or cached resource.

    Raises:
        (Potentially errors from `huggingface_hub.hf_hub_download` if the download
         fails, e.g., network issues, file not found in repo, etc.)
    """
    cache_dir = get_cache_dir(repo_id.replace("/", "_"))
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, force_download=force_download, cache_dir=str(cache_dir))


def get_cache_dir(name: str) -> pathlib.Path:
    """Get or create a standardized cache directory path for persistent storage.

    Constructs a path within the user's home directory under `.biasx/cache/`.
    Creates the directory (including parent directories) if it doesn't exist.
    Used by `get_resource_path` to determine where to store downloaded files.

    Args:
        name (str): A specific name for the subdirectory within the cache
                    (e.g., the `repo_id` with slashes replaced).

    Returns:
        A `pathlib.Path` object representing the absolute path to the
        specific cache subdirectory.
    """
    cache_dir = pathlib.Path(os.path.expanduser("~")) / ".biasx" / "cache" / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_file_path(caller_file: str, path: str) -> pathlib.Path:
    """Get the absolute path to a file relative to the calling module.

    Resolves a relative path based on the directory containing the Python file
    that calls this function. Checks if the resulting file path exists.

    Args:
        caller_file (str): The path to the Python file calling this function
                           (typically `__file__` from the caller).
        path (str): The relative path from the `caller_file`'s directory to the
                    target file (e.g., 'data/landmark_mapping.json').

    Returns:
        A `pathlib.Path` object representing the absolute path to the target file.

    Raises:
        FileNotFoundError: If the resolved file path does not exist.
    """
    module_dir = pathlib.Path(caller_file).parent
    file_path = module_dir / path

    if not file_path.exists():
        raise FileNotFoundError(f"File not found at {file_path}")

    return file_path
