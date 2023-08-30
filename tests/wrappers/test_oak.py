import logging
import shutil

import pytest
from oaklib import get_adapter

from curate_gpt import ChromaDBAdapter
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers.ontology.ontology_wrapper import OntologyWrapper
from tests import INPUT_DIR, OUTPUT_DIR

TEMP_OAKVIEW_DB = OUTPUT_DIR / "oaktmp"
TEMP_OAKVIEW_DB2 = OUTPUT_DIR / "oaktmp2"

# logger = logging.getLogger(__name__)

logging.basicConfig()
logger = logging.root
logger.setLevel(logging.DEBUG)


@pytest.fixture
def vstore() -> OntologyWrapper:
    adapter = get_adapter(INPUT_DIR / "go-nucleus.db")
    db = ChromaDBAdapter(str(TEMP_OAKVIEW_DB))
    db.reset()
    return OntologyWrapper(oak_adapter=adapter, local_store=db, extractor=BasicExtractor())


def test_oak_objects(vstore):
    shutil.rmtree(TEMP_OAKVIEW_DB, ignore_errors=True)
    # vstore.local_store.reset()
    objs = list(vstore.objects())
    [nucleus] = [obj for obj in objs if obj["id"] == "Nucleus"]
    assert nucleus["label"] == "nucleus"
    assert nucleus["original_id"] == "GO:0005634"


def test_oak_index(vstore):
    shutil.rmtree(TEMP_OAKVIEW_DB2, ignore_errors=True)
    adapter = get_adapter(INPUT_DIR / "go-nucleus.db")
    db = ChromaDBAdapter(str(TEMP_OAKVIEW_DB2))
    db.reset()
    wrapper = OntologyWrapper(oak_adapter=adapter, local_store=db, extractor=BasicExtractor())
    db.insert(wrapper.objects())
    g = wrapper.from_object(
        {
            "id": "Nucleus",
            "label": "nucleus",
            "relationships": [{"predicate": "rdfs:subClassOf", "target": "Organelle"}],
            "original_id": "GO:0005634",
        },
        store=db,
    )
    print(g.nodes)
    print(g.edges)


def test_oak_search(vstore):
    results = list(vstore.search("nucl"))
    assert len(results) > 0
    for result in results:
        print(result)
