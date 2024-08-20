from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict


class CollectionMetadata(BaseModel):
    """
    Metadata about a collection.

    This is an open class, so additional metadata can be added.
    """

    model_config = ConfigDict(protected_namespaces=())

    name: Optional[str] = None
    """Name of the collection"""

    description: Optional[str] = None
    """Description of the collection"""

    model: Optional[str] = None
    """Name of any ML model"""

    object_type: Optional[str] = None
    """Type of object in the collection"""

    source: Optional[str] = None
    """Source of the collection"""

    # DEPRECATED
    annotations: Optional[Dict] = None
    """Additional metadata"""

    object_count: Optional[int] = None
    """Number of objects in the collection"""

    hnsw_space: Optional[str] = None
    """Space used for hnsw index (e.g. 'cosine')"""
