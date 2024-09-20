import pytest

from curate_gpt import ChromaDBAdapter
from curate_gpt.store import SchemaProxy
from curate_gpt.wrappers.ontology import ONTOLOGY_MODEL_PATH
from tests import INPUT_DBS

# TODO: this has to be reviewed, isolate more, dont use one db for multiple tests
# - the current setup does not allow reset
# - set collection is v vulnerable, easier setting/creating new col for each test
# - using a loaded a test ontology can also be mocked for ease
# - ? use structure from tests/wrapper (vstore,wrapper fixtures)
# - or create collection in each test to use and load all collections with the whole data and reset/remove collection after
@pytest.fixture
def go_test_chroma_db() -> ChromaDBAdapter:
    """
    Fixture for a ChromaDBAdapter instance with the test ontology loaded.

    Note: the chromadb is not checked into github - instead,
    this relies on test_chromadb_adapter.test_store to create the test db.
    """
    db = ChromaDBAdapter(str(INPUT_DBS / "go-nucleus-chroma"))
    db.schema_proxy = SchemaProxy(ONTOLOGY_MODEL_PATH)
    db.set_collection("test")
    return db
