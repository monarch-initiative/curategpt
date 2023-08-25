import logging
import shutil
import time

import yaml

from curate_gpt import ChromaDBAdapter
from curate_gpt.agents.dac_agent import DatabaseAugmentedCompletion
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers.literature import WikipediaWrapper
from tests import OUTPUT_DIR

TEMP_Wikipedia_DB = OUTPUT_DIR / "wp_tmp"

logger = logging.getLogger(__name__)


def test_wikipedia_search():
    shutil.rmtree(TEMP_Wikipedia_DB, ignore_errors=True)
    db = ChromaDBAdapter(str(TEMP_Wikipedia_DB))
    extractor = BasicExtractor()
    db.reset()
    wikipedia = WikipediaWrapper(local_store=db, extractor=extractor)
    results = list(wikipedia.search("acinar cells of the salivary gland"))
    assert len(results) > 0
    for obj, _dist, _ in results:
        print(yaml.dump(obj))
    top_result = results[0][0]
    print(top_result)
    time.sleep(0.5)
    results2 = list(wikipedia.search(top_result["title"]))
    assert len(results2) > 0
