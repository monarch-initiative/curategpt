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
from .schema_proxy import SchemaProxy

__all__ = ["ChromaDBAdapter", "SchemaProxy", "DBAdapter"]
