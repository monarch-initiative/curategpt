from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional, TextIO

from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel

from curate_gpt.store.duckdb_result import DuckDBSearchResult

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

MODEL_DIMENSIONS = {"all-MiniLM-L6-v2": 384}
OPENAI_MODEL_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
MODELS = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]