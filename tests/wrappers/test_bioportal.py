import logging

import pytest
from curategpt.extract import BasicExtractor
from curategpt.wrappers.ontology.bioportal_wrapper import BioportalWrapper
from curategpt.wrappers.ontology.ontology_wrapper import OntologyWrapper

from tests import OUTPUT_DIR
from tests.utils.helper import DEBUG_MODE, create_db_dir, setup_db

TEMP_OAKVIEW_DB = OUTPUT_DIR / "bioportal_tmp"

# logger = logging.getLogger(__name__)

logging.basicConfig()
logger = logging.root
logger.setLevel(logging.DEBUG)


@pytest.fixture
def vstore(tmp_path) -> OntologyWrapper:
    tmp_dir = create_db_dir(tmp_path=tmp_path, out_dir=TEMP_OAKVIEW_DB)
    db = setup_db(tmp_dir)
    db.reset()
    try:
        view = BioportalWrapper(local_store=db, extractor=BasicExtractor())
        assert view.fetch_definitions is False
        yield view
    except Exception as e:
        raise e
    finally:
        if not DEBUG_MODE:
            db.reset()

    # view = BioportalView(oak_adapter=adapter, local_store=db, extractor=BasicExtractor())
    # view.fetch_definitions = False
    # view.fetch_relationships = False


@pytest.mark.skip(reason="OAK bp wrapper doesn't support definitions yets")
def test_oak_search(vstore):
    results = list(
        vstore.search(
            "what are the genes associated with Castleman disease and HHV8?",
            limit=50,
            external_search_limit=300,
        )
    )
    assert len(results) > 0
    for result in results:
        print(result)
