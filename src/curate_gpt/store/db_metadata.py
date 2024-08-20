from pathlib import Path

import yaml
from pydantic import BaseModel


class DBSettings(BaseModel):
    name: str = "duckdb"
    """Name of the database."""

    model: str = "all-MiniLM-L6-v2"
    """Name of any embedding model"""

    hnsw_space: str = "cosine"
    """Space used for hnsw index (e.g. 'cosine')."""

    ef_construction: int = 128
    """
    Construction parameter for hnsw index.
    Higher values are more accurate but slower.
    """

    ef_search: int = 64
    """
    Search parameter for hnsw index.
    Higher values are more accurate but slower.
    """

    M: int = 16
    """M parameter for hnsw index"""

    def load_config(self, path: Path):
        with open(path) as file:
            config = yaml.safe_load(file)
            self.name = config.get("name", self.name)
            self.hnsw_space = config.get("hnsw_space", self.hnsw_space)
            self.model = config.get("model", self.model)
            self.ef_construction = config.get("ef_construction", self.ef_construction)
            self.ef_search = config.get("ef_search", self.ef_search)
            self.M = config.get("M", self.M)
