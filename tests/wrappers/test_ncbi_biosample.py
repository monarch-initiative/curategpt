import logging
import shutil
import time

import yaml
from curate_gpt import ChromaDBAdapter
from curate_gpt.agents.chat_agent import ChatAgent
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers.investigation.ncbi_biosample_wrapper import NCBIBiosampleWrapper

from tests import OUTPUT_DIR

TEMP_BIOSAMPLE_DB = OUTPUT_DIR / "biosample_tmp"

logger = logging.getLogger(__name__)


def test_biosample_search():
    shutil.rmtree(TEMP_BIOSAMPLE_DB, ignore_errors=True)
    db = ChromaDBAdapter(str(TEMP_BIOSAMPLE_DB))
    extractor = BasicExtractor()
    db.reset()
    wrapper = NCBIBiosampleWrapper(local_store=db, extractor=extractor)
    results = list(wrapper.search("gut microbiome IBD Crohn's disease"))
    assert len(results) > 0
    top_result = results[0][0]
    print(yaml.dump(top_result))
    time.sleep(0.5)
    results2 = list(wrapper.search(top_result["title"]))
    assert len(results2) > 0


def test_biosample_chat():
    shutil.rmtree(TEMP_BIOSAMPLE_DB, ignore_errors=True)
    db = ChromaDBAdapter(str(TEMP_BIOSAMPLE_DB))
    extractor = BasicExtractor()
    db.reset()
    wrapper = NCBIBiosampleWrapper(local_store=db, extractor=extractor)
    chat = ChatAgent(knowledge_source=wrapper, extractor=extractor)
    response = chat.chat("what are some characteristics of the gut microbiome in Crohn's disease?")
    print(response.formatted_body)
    for ref in response.references:
        print(ref)
