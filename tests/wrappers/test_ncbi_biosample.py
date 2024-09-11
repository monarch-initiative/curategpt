import logging
import os
import shutil
import tempfile
import time

import yaml

from curate_gpt import ChromaDBAdapter
from curate_gpt.agents.chat_agent import ChatAgent
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers.investigation.ncbi_biosample_wrapper import NCBIBiosampleWrapper
from tests import OUTPUT_DIR
from tests.store.conftest import requires_openai_api_key

TEMP_BIOSAMPLE_DB = OUTPUT_DIR / "biosample_tmp"

logger = logging.getLogger(__name__)


@requires_openai_api_key
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


@requires_openai_api_key
def test_biosample_chat():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, TEMP_BIOSAMPLE_DB)
        # shutil.rmtree(TEMP_BIOSAMPLE_DB, ignore_errors=True)
        db = ChromaDBAdapter(db_path)
        extractor = BasicExtractor()
        db.reset()
        wrapper = NCBIBiosampleWrapper(local_store=db, extractor=extractor)
        chat = ChatAgent(knowledge_source=wrapper, extractor=extractor)
        response = chat.chat("what are some characteristics of the gut microbiome in Crohn's disease?")
        print(response.formatted_body)
        for ref in response.references:
            print(ref)
