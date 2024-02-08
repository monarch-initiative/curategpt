import functools
import os
from pathlib import Path
from typing import TypedDict

import toml

CWD = Path.cwd()
SETTINGS_PATH = os.getenv("AZURE_SETTINGS_PATH", None)
SETTINGS_NAME = os.getenv("AZURE_SETTINGS_NAME", "azure")
USE_AZURE = os.getenv("USE_AZURE", "false") == "true"


def read_toml(path: Path) -> dict:
    with path.open() as cf:
        return toml.load(cf)


class ModelSettings(TypedDict):
    api_key: str
    api_version: str
    base_url: str
    deployment_name: str
    model_name: str


class AzureSettings(TypedDict):
    chat_model: ModelSettings
    embedding_model: ModelSettings


@functools.lru_cache(maxsize=1)
def get_azure_settings(base_dir: Path = CWD) -> AzureSettings:
    etc_dir = base_dir / "etc"
    default_settings = etc_dir / f"{SETTINGS_NAME}.toml"
    settings_override = etc_dir / f"{SETTINGS_NAME}_custom.toml"
    if SETTINGS_PATH:
        settings_path = Path(SETTINGS_PATH)
    elif settings_override.is_file():
        settings_path = settings_override
    else:
        settings_path = default_settings
    return read_toml(settings_path) if settings_path.is_file() else {}
