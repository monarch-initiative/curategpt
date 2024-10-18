import logging
import time

import pytest

from curategpt.agents.chat_agent import ChatAgent
from curategpt.extract import BasicExtractor
from curategpt.wrappers.literature import PubmedWrapper
from tests import OUTPUT_DIR
from tests.store.conftest import requires_openai_api_key
from tests.utils.helper import DEBUG_MODE, create_db_dir, setup_db

TEMP_PUBMED_SEARCH = OUTPUT_DIR / "pmid_tmp"
TEMP_PUBMED_CHAT = OUTPUT_DIR / "pmid_tmp"

# logger = logging.getLogger(__name__)

logging.basicConfig()
logger = logging.root
logger.setLevel(logging.DEBUG)


@pytest.fixture
def wrapper(request, tmp_path):
    db = None
    if hasattr(request, "param"):
        tmp_base = request.param
        temp_dir = create_db_dir(tmp_path, tmp_base)
        db = setup_db(temp_dir)
    extractor = BasicExtractor()
    try:
        pubmed = PubmedWrapper(extractor=extractor)
        yield pubmed
    except Exception as e:
        raise e
    finally:
        if not DEBUG_MODE and db is not None:
            db.reset()


def test_pubmed_by_id(wrapper):
    objs = wrapper.objects_by_ids(["12754706"])
    assert len(objs) == 1


def test_pubmed_to_pmc(wrapper):
    pmcid = wrapper.fetch_pmcid("PMID:35663206")
    assert pmcid == "PMC:PMC9159873"


def test_full_text(wrapper):
    txt = wrapper.fetch_full_text("PMID:35663206")
    print(len(txt))
    print(txt[0:100])
    print(txt)


@requires_openai_api_key
@pytest.mark.parametrize("wrapper", [TEMP_PUBMED_SEARCH], indirect=True)
def test_pubmed_search(wrapper):
    results = list(wrapper.search("acinar cells of the salivary gland"))
    assert len(results) > 0
    top_result = results[0][0]
    print(top_result)
    time.sleep(0.5)
    results2 = list(wrapper.search(top_result["title"]))
    assert len(results2) > 0


@requires_openai_api_key
@pytest.mark.parametrize("wrapper", [TEMP_PUBMED_CHAT], indirect=True)
def test_pubmed_chat(wrapper):
    chat = ChatAgent(knowledge_source=wrapper, extractor=wrapper.extractor)
    response = chat.chat("what diseases are associated with acinar cells of the salivary gland")
    print(response)
