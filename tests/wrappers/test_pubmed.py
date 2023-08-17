import logging
import shutil
import time

from curate_gpt import ChromaDBAdapter
from curate_gpt.agents.dae_agent import DatabaseAugmentedExtractor
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers import PubmedWrapper
from tests import OUTPUT_DIR

TEMP_PUBMED_DB = OUTPUT_DIR / "pmid_tmp"

# logger = logging.getLogger(__name__)

logging.basicConfig()
logger = logging.root
logger.setLevel(logging.DEBUG)


def test_pubmed_search():
    shutil.rmtree(TEMP_PUBMED_DB, ignore_errors=True)
    db = ChromaDBAdapter(str(TEMP_PUBMED_DB))
    extractor = BasicExtractor()
    db.reset()
    pubmed = PubmedWrapper(local_store=db, extractor=extractor)
    results = list(pubmed.search("acinar cells of the salivary gland"))
    assert len(results) > 0
    top_result = results[0][0]
    print(top_result)
    time.sleep(0.5)
    results2 = list(pubmed.search(top_result["title"]))
    assert len(results2) > 0
    dalek = DatabaseAugmentedExtractor(knowledge_source=db, extractor=extractor)
    ao = dalek.generate_extract(
        "the role of acinar cells of the salivary gland in disease", context_property="title"
    )
    print(ao.object)


def test_pubmed_chat():
    shutil.rmtree(TEMP_PUBMED_DB, ignore_errors=True)
    db = ChromaDBAdapter(str(TEMP_PUBMED_DB))
    extractor = BasicExtractor()
    db.reset()
    pubmed = PubmedWrapper(local_store=db, extractor=extractor)
    response = pubmed.chat("what diseases are associated with acinar cells of the salivary gland")
    print(response)
