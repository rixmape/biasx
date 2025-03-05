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
    """Load a JSON configuration file from the module's data directory."""
    module_dir = pathlib.Path(caller_file).parent
    data_path = module_dir / "data" / config_file

    if not data_path.exists():
        raise FileNotFoundError(f"File not found at {data_path}")

    with open(data_path, "r") as f:
        return json.load(f)


@lru_cache(maxsize=64)
def get_resource_path(repo_id: str, filename: str, repo_type: str = "dataset", force_download: bool = False) -> str:
    """Download or retrieve a cached resource from HuggingFace Hub."""
    cache_dir = get_cache_dir(repo_id.replace("/", "_"))
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, force_download=force_download, cache_dir=str(cache_dir))


def get_cache_dir(name: str) -> pathlib.Path:
    """Get a standardized cache directory for persistent storage."""
    cache_dir = pathlib.Path(os.path.expanduser("~")) / ".biasx" / "cache" / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_file_path(caller_file: str, path: str) -> pathlib.Path:
    """Get path to a file relative to a module."""
    module_dir = pathlib.Path(caller_file).parent
    file_path = module_dir / path

    if not file_path.exists():
        raise FileNotFoundError(f"File not found at {file_path}")

    return file_path
