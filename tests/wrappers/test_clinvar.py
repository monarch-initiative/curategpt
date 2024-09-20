import logging
import time

import pytest
import requests
import yaml

from curate_gpt.agents.chat_agent import ChatAgent
from curate_gpt.agents.dragon_agent import DragonAgent
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers.clinical.clinvar_wrapper import ClinVarWrapper
from tests import INPUT_DIR, OUTPUT_DIR
from tests.store.conftest import requires_openai_api_key
from tests.utils.helper import DEBUG_MODE, create_db_dir, setup_db

TEMP_DB = OUTPUT_DIR / "obj_tmp"


logger = logging.getLogger(__name__)


# TODO: add this example file
@pytest.mark.skip(reason="This test requires a specific example file")
def test_clinvar_transform():
    obj = yaml.safe_load(open(INPUT_DIR / "clinvar-esummary-example.yaml"))
    wrapper = ClinVarWrapper()
    vars = wrapper.objects_from_dict(obj)
    print(yaml.dump(vars))


@pytest.fixture
def wrapper(tmp_path) -> ClinVarWrapper:
    temp_dir = create_db_dir(tmp_path, TEMP_DB)
    db = setup_db(temp_dir)
    extractor = BasicExtractor()
    try:
        yield ClinVarWrapper(local_store=db, extractor=extractor)
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error occurred: {e}")
        raise e
    finally:
        if not DEBUG_MODE:
            db.reset()


@requires_openai_api_key
def test_clinvar_search(wrapper):
    results = list(wrapper.search("IBD Crohn's disease and related diseases"))
    assert len(results) > 0
    top_result = results[0][0]
    print(yaml.dump(top_result))
    time.sleep(0.5)
    results2 = list(wrapper.search(top_result["title"]))
    assert len(results2) > 0


@requires_openai_api_key
def test_clinvar_chat(wrapper):
    extractor = DragonAgent()
    extractor = BasicExtractor()
    chat = ChatAgent(knowledge_source=wrapper, extractor=extractor)
    response = chat.chat("what are the major variants and genes underpinning Crohn's disease?")
    print(response.formatted_body)
    for ref in response.references:
        print(ref)
