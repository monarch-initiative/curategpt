import logging
import time

import pytest
import yaml

from curategpt import ChromaDBAdapter
from curategpt.agents.chat_agent import ChatAgent
from curategpt.extract import BasicExtractor
from curategpt.wrappers.literature.pmc_wrapper import PMCWrapper
from tests import INPUT_DIR, OUTPUT_DIR
from tests.utils.helper import DEBUG_MODE, create_db_dir

TEMP_PMC_SEARCH = OUTPUT_DIR / "pmc_search"
TEMP_PMC_CHAT = OUTPUT_DIR / "pmc_chat"

logger = logging.getLogger(__name__)


@pytest.mark.skip("TODO")
def test_pmc_transform():
    obj = yaml.safe_load(open(INPUT_DIR / "pmc-fetch-example.yaml"))
    wrapper = PMCWrapper()
    vars = wrapper.objects_from_dict(obj)
    print(yaml.dump(vars))


@pytest.fixture
def wrapper(request, tmp_path) -> PMCWrapper:
    temp_base = request.param
    temp_db = create_db_dir(tmp_path, temp_base)
    db = ChromaDBAdapter(temp_db)
    extractor = BasicExtractor()
    try:
        PMCWrapper(local_store=db, extractor=extractor)
    except Exception as e:
        raise e
    finally:
        if not DEBUG_MODE:
            db.reset()




@pytest.mark.skip("TODO")
@pytest.mark.parametrize("wrapper", [TEMP_PMC_SEARCH], indirect=True)
def test_pmc_search(wrapper):
    results = list(wrapper.search("IBD Crohn's disease and related diseases"))
    assert len(results) > 0
    top_result = results[0][0]
    print(yaml.dump(top_result))
    time.sleep(0.5)
    results2 = list(wrapper.search(top_result["title"]))
    assert len(results2) > 0


@pytest.mark.skip("TODO")
@pytest.mark.parametrize("wrapper", [TEMP_PMC_CHAT], indirect=True)
def test_pmc_chat(wrapper):
    # extractor = DragonAgent()
    chat = ChatAgent(knowledge_source=wrapper, extractor=wrapper.extractor)
    response = chat.chat("what are the major variants and genes underpinning Crohn's disease?")
    print(response.formatted_body)
    for ref in response.references:
        print(ref)
