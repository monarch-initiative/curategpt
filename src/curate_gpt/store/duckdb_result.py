import json

from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class DuckDBSearchResult(BaseModel):
    id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embeddings: Optional[List[float]] = None
    documents: Optional[str] = None
    distance: Optional[float] = 0  # TODO: technically this is cosim similarity but for now we'll call it distance

    def __repr__(self):
        return json.dumps({
            "id": self.id,
            "metadata": self.metadata,
            "embeddings": self.embeddings,
            "documents": self.documents,
            "distance": self.distance
        }, indent=2)
