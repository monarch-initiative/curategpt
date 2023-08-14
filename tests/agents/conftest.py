import pytest

from curate_gpt import ChromaDBAdapter
from curate_gpt.store import SchemaProxy
from curate_gpt.view import ONTOLOGY_MODEL_PATH
from tests import INPUT_DBS


@pytest.fixture
def go_test_chroma_db() -> ChromaDBAdapter:
    db = ChromaDBAdapter(str(INPUT_DBS / "go-nucleus-chroma"))
    db.schema_proxy = SchemaProxy(ONTOLOGY_MODEL_PATH)
    return db
