"""Small .env loader for local experiment commands.

This avoids adding another dependency just to pick up HF_TOKEN from the
workspace root. Existing environment variables win.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_from_parents(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        env_path = directory / ".env"
        if env_path.exists():
            _load_env_file(env_path)
            return env_path
    return None


def _load_env_file(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
