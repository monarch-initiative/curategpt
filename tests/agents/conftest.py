import pytest

from curate_gpt import ChromaDBAdapter
from curate_gpt.store import SchemaProxy
from curate_gpt.wrappers.ontology import ONTOLOGY_MODEL_PATH
from tests import INPUT_DBS


@pytest.fixture
def go_test_chroma_db() -> ChromaDBAdapter:
    """
    Fixture for a ChromaDBAdapter instance with the test ontology loaded.

    Note: the chromadb is not checked into github - instead,
    this relies on test_chromadb_dapter.test_store to create the test db.
    """
    db = ChromaDBAdapter(str(INPUT_DBS / "go-nucleus-chroma"))
    db.schema_proxy = SchemaProxy(ONTOLOGY_MODEL_PATH)
    db.set_collection("test")
    return db
