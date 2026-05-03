"""YAML config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def dtype_name(config: dict[str, Any]) -> str:
    return str(config.get("model", {}).get("torch_dtype", "bfloat16"))


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16
