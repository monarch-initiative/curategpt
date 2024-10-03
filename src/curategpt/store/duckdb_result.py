from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict

SEARCH_RESULT = Tuple[Dict[str, Any], Dict, float, Optional[Dict]]


class DuckDBSearchResult(BaseModel):

    model_config = ConfigDict(protected_namespaces=())

    ids: Optional[str] = None
    metadatas: Optional[Dict[str, Any]] = None
    embeddings: Optional[List[float]] = None
    documents: Optional[str] = None
    distances: Optional[float] = 0  # TODO: technically this is simple cosim similarity
    include: Optional[Set[str]] = None

    def to_json(self, indent: int = 2):
        return self.model_dump_json(include=self.include, indent=indent)

    def to_dict(self):
        if self.include:
            return self.model_dump(include=self.include)
        return self.model_dump()

    def __repr__(self, indent: int = 2):
        return self.model_dump_json(include=self.include, indent=indent)

    def __iter__(self) -> Iterator[SEARCH_RESULT]:
        # TODO vocab.py for 'VARIABLES', but circular import
        metadata_include = "metadatas" in self.include if self.include else True
        embeddings_include = "embeddings" in self.include if self.include else True
        documents_include = "documents" in self.include if self.include else True
        similarity_include = "distances" in self.include if self.include else True

        obj = self.metadatas if metadata_include else {}
        meta = {
            "_embeddings": self.embeddings if embeddings_include else None,
            "documents": self.documents if documents_include else None,
        }
        distance = self.distances if similarity_include else None

        yield obj, distance, meta
