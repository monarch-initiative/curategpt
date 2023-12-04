import logging
import shutil
import time

import pytest
import yaml
from curate_gpt import ChromaDBAdapter
from curate_gpt.agents.chat_agent import ChatAgent
from curate_gpt.agents.dragon_agent import DragonAgent
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers.clinical.clinvar_wrapper import ClinVarWrapper

from tests import INPUT_DIR, OUTPUT_DIR

TEMP_DB = OUTPUT_DIR / "obj_tmp"


logger = logging.getLogger(__name__)


def test_clinvar_transform():
    obj = yaml.safe_load(open(INPUT_DIR / "clinvar-esummary-example.yaml"))
    wrapper = ClinVarWrapper()
    vars = wrapper.objects_from_dict(obj)
    print(yaml.dump(vars))


@pytest.fixture
def wrapper() -> ClinVarWrapper:
    shutil.rmtree(TEMP_DB, ignore_errors=True)
    db = ChromaDBAdapter(str(TEMP_DB))
    extractor = BasicExtractor()
    db.reset()
    return ClinVarWrapper(local_store=db, extractor=extractor)


def test_clinvar_search(wrapper):
    results = list(wrapper.search("IBD Crohn's disease and related diseases"))
    assert len(results) > 0
    top_result = results[0][0]
    print(yaml.dump(top_result))
    time.sleep(0.5)
    results2 = list(wrapper.search(top_result["title"]))
    assert len(results2) > 0


def test_clinvar_chat(wrapper):
    extractor = DragonAgent()
    extractor = BasicExtractor()
    chat = ChatAgent(knowledge_source=wrapper, extractor=extractor)
    response = chat.chat("what are the major variants and genes underpinning Crohn's disease?")
    print(response.formatted_body)
    for ref in response.references:
        print(ref)
