"""Utility functions and common operations for the BiasX library."""

import json
import os
import pathlib
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional, Type, TypeVar, Union

from huggingface_hub import hf_hub_download

T = TypeVar("T", bound=Enum)


def get_module_data_path(caller_file: str, data_file: str) -> pathlib.Path:
    """Get path to a data file relative to a module."""
    module_dir = pathlib.Path(caller_file).parent
    data_path = module_dir / "data" / data_file

    if not data_path.exists():
        raise FileNotFoundError(f"File not found at {data_path}")

    return data_path


@lru_cache(maxsize=32)
def load_json_config(caller_file: str, config_file: str) -> Dict[str, Any]:
    """Load a JSON configuration file from the module's data directory."""
    config_path = get_module_data_path(caller_file, config_file)

    with open(config_path, "r") as f:
        return json.load(f)


def download_resource(
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    force_download: bool = False,
    cache_dir: Optional[str] = None,
) -> str:
    """Download a resource from HuggingFace Hub."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        force_download=force_download,
        cache_dir=cache_dir,
    )


def parse_enum(value: Optional[Union[str, T]], enum_class: Type[T], default: Optional[T] = None) -> Optional[T]:
    """Convert a string to an enum value."""
    if value is None:
        return default

    if isinstance(value, enum_class):
        return value

    try:
        return enum_class(value)
    except (ValueError, TypeError):
        try:
            return enum_class[str(value)]
        except (KeyError, TypeError):
            return default


def get_cache_dir(name: str) -> pathlib.Path:
    """Get a standardized cache directory for persistent storage."""
    cache_dir = pathlib.Path(os.path.expanduser("~")) / ".biasx" / "cache" / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
