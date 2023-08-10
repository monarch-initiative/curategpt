"""curate-gpt package."""
import importlib_metadata

try:
    __version__ = importlib_metadata.version(__name__)
except importlib_metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"  # pragma: no cover

from curate_gpt.store import ChromaDBAdapter, DBAdapter
from curate_gpt.view.ontology_view import OntologyView

__all__ = ["ChromaDBAdapter", "DBAdapter", "OntologyView"]
