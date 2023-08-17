from typing import Dict, Optional

from pydantic import BaseModel


class CollectionMetadata(BaseModel):
    """
    Metadata about a collection.

    This is an open class, so additional metadata can be added.
    """

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
