from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List

from .config import DatasetSpec, ExperimentSpec, ModelSpec, PromptSpec, ToolConfig


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


VALID_PROVIDERS = {"openai", "anthropic", "google", "hf", "local"}
VALID_DATASET_KINDS = {"smartbugs_curated", "solidifi", "benign_contracts", "custom"}


def validate_config(cfg: Dict, config_dir: Path) -> None:
    """Validate experiment configuration.

    Args:
        cfg: Parsed YAML config dict
        config_dir: Directory containing the config file (for relative paths)

    Raises:
        ConfigValidationError: If validation fails
    """
    errors = []

    # Required top-level fields
    required_fields = ["experiment_id", "prompts", "models", "datasets", "output_dir"]
    for field in required_fields:
        if field not in cfg:
            errors.append(f"Missing required field: {field}")

    if errors:
        raise ConfigValidationError("\n".join(errors))

    # Validate prompts
    for i, p in enumerate(cfg.get("prompts", [])):
        if "id" not in p:
            errors.append(f"prompts[{i}]: missing 'id' field")
        if "template_path" not in p:
            errors.append(f"prompts[{i}]: missing 'template_path' field")

    # Validate models
    for i, m in enumerate(cfg.get("models", [])):
        if "name" not in m:
            errors.append(f"models[{i}]: missing 'name' field")
        if "provider" not in m:
            errors.append(f"models[{i}]: missing 'provider' field")
        elif m["provider"] not in VALID_PROVIDERS:
            errors.append(
                f"models[{i}]: unknown provider '{m['provider']}'. "
                f"Valid providers: {', '.join(sorted(VALID_PROVIDERS))}"
            )
        if "params" not in m:
            errors.append(f"models[{i}]: missing 'params' field")

    # Validate datasets
    for i, d in enumerate(cfg.get("datasets", [])):
        if "name" not in d:
            errors.append(f"datasets[{i}]: missing 'name' field")
        if "kind" not in d:
            errors.append(f"datasets[{i}]: missing 'kind' field")
        if "path" not in d:
            errors.append(f"datasets[{i}]: missing 'path' field")

    if errors:
        raise ConfigValidationError("\n".join(errors))


def load_experiment_config(path: str | Path, validate: bool = True) -> ExperimentSpec:
    """Load and optionally validate experiment configuration from YAML file.

    Args:
        path: Path to the YAML config file
        validate: Whether to validate the config (default True)

    Returns:
        ExperimentSpec with loaded configuration

    Raises:
        ConfigValidationError: If validation is enabled and config is invalid
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    if validate:
        validate_config(cfg, path.parent)

    prompts = [PromptSpec(**p) for p in cfg["prompts"]]
    models = [ModelSpec(**m) for m in cfg["models"]]
    datasets = [DatasetSpec(**d) for d in cfg["datasets"]]

    # Parse tool configuration (optional)
    tools = None
    if "tools" in cfg and cfg["tools"]:
        tools = ToolConfig(**cfg["tools"])

    return ExperimentSpec(
        experiment_id=cfg["experiment_id"],
        random_seed=int(cfg.get("random_seed", 0)),
        runs_per_item=int(cfg.get("runs_per_item", 1)),
        prompts=prompts,
        models=models,
        datasets=datasets,
        output_dir=cfg["output_dir"],
        notes=cfg.get("notes", ""),
        tools=tools,
    )
