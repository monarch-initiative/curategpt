import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DuckDBSearchResult(BaseModel):
    id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embeddings: Optional[List[float]] = None
    documents: Optional[str] = None
    distance: Optional[float] = (
        0  # TODO: technically this is cosim similarity but for now we'll call it distance
    )

    def __repr__(self):
        return json.dumps(
            {
                "id": self.id,
                "metadata": self.metadata,
                "embeddings": self.embeddings,
                "documents": self.documents,
                "distance": self.distance,
            },
            indent=2,
        )


# TODO: check if this is valid
# this could be the potentially future DuckDBSearchResult allowing to have minimal changes in the codebase
# as return type of ChromaDB and DuckDB would be equivalent
class PotentialFutureDuckDBResult(BaseModel):
    id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embeddings: Optional[List[float]] = None
    documents: Optional[str] = None
    distance: Optional[float] = 0  # technically cosine similarity (is handled in return)

    def __iter__(self):
        """
            Allow unpacking directly into (obj, distance, meta).
            """
        obj = {
            "metadata": self.metadata,
        }
        meta = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "documents": self.documents,

        }
        return iter((obj, (1 - self.distance), meta))
