import pytest
from curategpt import ChromaDBAdapter
from curategpt.store import SchemaProxy
from curategpt.wrappers.ontology import ONTOLOGY_MODEL_PATH, OntologyWrapper
from oaklib import get_adapter

from tests import INPUT_DBS, INPUT_DIR


@pytest.fixture
def ontology_db() -> ChromaDBAdapter:
    db = ChromaDBAdapter(str(INPUT_DBS / "go-nucleus-chroma"))
    db.schema_proxy = SchemaProxy(ONTOLOGY_MODEL_PATH)
    db.client.reset()
    # db.default_model = "openai:"
    return db


@pytest.fixture
def loaded_ontology_db(ontology_db) -> ChromaDBAdapter:
    db = ontology_db
    adapter = get_adapter(str(INPUT_DIR / "go-nucleus.db"))
    # ontology_db.linkml_schema_path = ONTOLOGY_MODEL_PATH
    view = OntologyWrapper(oak_adapter=adapter)
    ontology_db.text_lookup = view.text_field
    ontology_db.insert(view.objects(), collection="terms_go")
    ontology_db.text_lookup = "label"
    return db
