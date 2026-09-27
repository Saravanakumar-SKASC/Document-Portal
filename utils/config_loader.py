import os
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=4)
def load_config(config_path: str | None = None) -> dict:
    """
    Load the YAML config.

    Resolution order: explicit argument -> CONFIG_PATH env var -> config/config.yaml
    next to the project root. Paths are built with pathlib, so this works the same
    on macOS, Linux and Windows regardless of the current working directory.
    """
    path = Path(config_path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)
