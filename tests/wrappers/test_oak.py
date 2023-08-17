import logging
import shutil

import pytest
from oaklib import get_adapter

from curate_gpt import ChromaDBAdapter
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers.ontology.ontology_wrapper import OntologyWrapper
from tests import INPUT_DIR, OUTPUT_DIR

TEMP_OAKVIEW_DB = OUTPUT_DIR / "oaktmp"

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
    for obj in vstore.objects():
        print(obj)


def test_oak_search(vstore):
    results = list(vstore.search("nucl"))
    assert len(results) > 0
    for result in results:
        print(result)
