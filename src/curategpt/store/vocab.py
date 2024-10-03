from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple, Union

from curate_gpt.store.duckdb_result import DuckDBSearchResult
from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel

OBJECT = Union[YAMLRoot, BaseModel, Dict, DuckDBSearchResult]
QUERY = Union[str, YAMLRoot, BaseModel, Dict, DuckDBSearchResult]
PROJECTION = Union[str, List[str]]
DEFAULT_COLLECTION = "test_collection"
SEARCH_RESULT = Tuple[DuckDBSearchResult, Dict, float, Optional[Dict]]
FILE_LIKE = Union[str, TextIO, Path]

IDS = "ids"
METADATAS = "metadatas"
EMBEDDINGS = "embeddings"
DOCUMENTS = "documents"
DISTANCES = "distances"

MODEL_MAP = {
    "text-embedding-ada-002": ("ada-002", 1536),
    "text-embedding-3-small": ("3-small", 1536),
    "text-embedding-3-large": ("3-large", 3072),
    "text-embedding-3-small-512": ("3-small-512", 512),
    "text-embedding-3-large-256": ("3-large-256", 256),
    "text-embedding-3-large-1024": ("3-large-1024", 1024),
}

DEFAULT_OPENAI_MODEL = "text-embedding-ada-002"
DEFAULT_MODEL = {"all-MiniLM-L6-v2": 384}
