"""Adapters for different document stores.

* Base class: :class:`DBAdapter`

Currently only one implementation is provided, for ChromaDB.
In future this will index

- MongoDB
- ElasticSearch
- Solr
- Postgres
- SQLite

Note: this package may become an independent project called linkml-store
in the future.
"""

from .chromadb_adapter import ChromaDBAdapter
from .db_adapter import DBAdapter
from .metadata import CollectionMetadata
from .schema_proxy import SchemaProxy

__all__ = ["DBAdapter", "ChromaDBAdapter", "SchemaProxy", "CollectionMetadata", "get_store"]


def get_all_subclasses(cls):
    """Recursively get all subclasses of a given class."""
    direct_subclasses = cls.__subclasses__()
    return direct_subclasses + [
        s for subclass in direct_subclasses for s in get_all_subclasses(subclass)
    ]


def get_store(name: str, *args, **kwargs) -> DBAdapter:
    from .in_memory_adapter import InMemoryAdapter  # noqa F401

    # noqa I005

    for c in get_all_subclasses(DBAdapter):
        if c.name == name:
            return c(*args, **kwargs)
    raise ValueError(f"Unknown view {name}, not found in {get_all_subclasses(DBAdapter)}")
