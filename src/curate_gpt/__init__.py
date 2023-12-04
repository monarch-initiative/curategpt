"""
CurateGPT: A framework semi-assisted curation of knowledge bases.

Architecture
============

* :mod:`.store`: json object stores that allow for embedding based search
* :mod:`.wrappers`: wraps external APIs and data sources for ingest
* :mod:`.extract`: extraction of json objects from LLMs
* :mod:`.agents`: agents that chain together search and generate components
* :mod:`.formatters`: formats data objects for presentation to humans and machine agents
* :mod:`.app`: streamlit application


"""
import importlib_metadata

try:
    __version__ = importlib_metadata.version(__name__)
except importlib_metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"  # pragma: no cover

from curate_gpt.extract import BasicExtractor, Extractor
from curate_gpt.store import ChromaDBAdapter, DBAdapter

__all__ = ["DBAdapter", "ChromaDBAdapter", "Extractor", "BasicExtractor"]
