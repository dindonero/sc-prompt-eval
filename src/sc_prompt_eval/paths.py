"""
Centralized path discovery for package data files.

This module provides robust path discovery that works in multiple contexts:
1. Running from repository checkout (development mode)
2. Installed via `pip install -e .` (editable install)
3. Installed via `pip install .` (regular install with package data)

The discovery strategy tries multiple locations in order of preference.
"""

from pathlib import Path
from typing import Optional
import os

# Cache discovered paths
_repo_root: Optional[Path] = None
_data_dir: Optional[Path] = None
_prompts_dir: Optional[Path] = None


def _find_repo_root() -> Optional[Path]:
    """Find the repository root by looking for marker files."""
    global _repo_root
    if _repo_root is not None:
        return _repo_root

    # Strategy 1: Walk up from this file looking for pyproject.toml
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Max 10 levels up
        if (current / "pyproject.toml").exists():
            _repo_root = current
            return _repo_root
        if (current / "prompts").is_dir() and (current / "data").is_dir():
            _repo_root = current
            return _repo_root
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Strategy 2: Check current working directory
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        _repo_root = cwd
        return _repo_root
    if (cwd / "prompts").is_dir() and (cwd / "data").is_dir():
        _repo_root = cwd
        return _repo_root

    # Strategy 3: Check environment variable
    env_root = os.environ.get("SC_PROMPT_EVAL_ROOT")
    if env_root:
        env_path = Path(env_root)
        if env_path.is_dir():
            _repo_root = env_path
            return _repo_root

    return None


def get_data_dir() -> Path:
    """Get the data directory path.

    Tries multiple locations:
    1. Repository root / data (development mode)
    2. Package directory / data (if data is shipped with package)
    3. SC_PROMPT_EVAL_DATA environment variable

    Returns:
        Path to the data directory

    Raises:
        FileNotFoundError: If data directory cannot be found
    """
    global _data_dir
    if _data_dir is not None and _data_dir.exists():
        return _data_dir

    # Strategy 1: Environment variable override
    env_data = os.environ.get("SC_PROMPT_EVAL_DATA")
    if env_data:
        env_path = Path(env_data)
        if env_path.is_dir():
            _data_dir = env_path
            return _data_dir

    # Strategy 2: Repository root
    repo_root = _find_repo_root()
    if repo_root:
        data_path = repo_root / "data"
        if data_path.is_dir():
            _data_dir = data_path
            return _data_dir

    # Strategy 3: Relative to package (if data shipped with package)
    pkg_data = Path(__file__).parent / "data"
    if pkg_data.is_dir():
        _data_dir = pkg_data
        return _data_dir

    # Strategy 4: Current working directory
    cwd_data = Path.cwd() / "data"
    if cwd_data.is_dir():
        _data_dir = cwd_data
        return _data_dir

    raise FileNotFoundError(
        "Could not find data directory. Tried:\n"
        f"  - SC_PROMPT_EVAL_DATA env var\n"
        f"  - Repository root / data\n"
        f"  - Package directory / data\n"
        f"  - Current working directory / data\n"
        "Set SC_PROMPT_EVAL_DATA environment variable or run from repository root."
    )


def get_prompts_dir() -> Path:
    """Get the prompts/templates directory path.

    Tries multiple locations:
    1. Repository root / prompts (development mode)
    2. Package directory / prompts/templates (if shipped with package)
    3. SC_PROMPT_EVAL_PROMPTS environment variable

    Returns:
        Path to the prompts directory

    Raises:
        FileNotFoundError: If prompts directory cannot be found
    """
    global _prompts_dir
    if _prompts_dir is not None and _prompts_dir.exists():
        return _prompts_dir

    # Strategy 1: Environment variable override
    env_prompts = os.environ.get("SC_PROMPT_EVAL_PROMPTS")
    if env_prompts:
        env_path = Path(env_prompts)
        if env_path.is_dir():
            _prompts_dir = env_path
            return _prompts_dir

    # Strategy 2: Repository root
    repo_root = _find_repo_root()
    if repo_root:
        prompts_path = repo_root / "prompts"
        if prompts_path.is_dir():
            _prompts_dir = prompts_path
            return _prompts_dir

    # Strategy 3: Relative to package (if prompts shipped with package)
    pkg_prompts = Path(__file__).parent / "prompts" / "templates"
    if pkg_prompts.is_dir():
        _prompts_dir = pkg_prompts
        return _prompts_dir

    # Strategy 4: Current working directory
    cwd_prompts = Path.cwd() / "prompts"
    if cwd_prompts.is_dir():
        _prompts_dir = cwd_prompts
        return _prompts_dir

    raise FileNotFoundError(
        "Could not find prompts directory. Tried:\n"
        f"  - SC_PROMPT_EVAL_PROMPTS env var\n"
        f"  - Repository root / prompts\n"
        f"  - Package directory / prompts/templates\n"
        f"  - Current working directory / prompts\n"
        "Set SC_PROMPT_EVAL_PROMPTS environment variable or run from repository root."
    )


def get_data_file(filename: str) -> Path:
    """Get path to a specific data file.

    Args:
        filename: Name of the file (e.g., "icl_examples.json")

    Returns:
        Full path to the data file

    Raises:
        FileNotFoundError: If the file cannot be found
    """
    data_dir = get_data_dir()
    file_path = data_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return file_path


def get_patterns_database_path() -> Path:
    """Get path to patterns_database.json."""
    return get_data_file("patterns_database.json")


def get_icl_examples_path() -> Path:
    """Get path to icl_examples.json."""
    return get_data_file("icl_examples.json")


def get_cot_cache_path() -> Path:
    """Get path to cot_cache.json (may not exist yet)."""
    return get_data_dir() / "cot_cache.json"
