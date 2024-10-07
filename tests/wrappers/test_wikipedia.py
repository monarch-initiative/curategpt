import logging
import time

import pytest
import yaml

from curategpt.extract import BasicExtractor
from curategpt.wrappers.literature import WikipediaWrapper
from tests import OUTPUT_DIR
from tests.store.conftest import requires_openai_api_key
from tests.utils.helper import DEBUG_MODE, create_db_dir, setup_db

TEMP_Wikipedia_DB = OUTPUT_DIR / "wp_tmp"

logger = logging.getLogger(__name__)

@pytest.fixture
def wrapper(request, tmp_path):
    db = None
    if hasattr(request, 'param'):
        tmp_base = request.param
        temp_dir = create_db_dir(tmp_path, tmp_base)
        db = setup_db(temp_dir)
    extractor = BasicExtractor()
    try:
        wiki = WikipediaWrapper(extractor=extractor)
        yield wiki
    except Exception as e:
        raise e
    finally:
        if not DEBUG_MODE and db is not None:
            db.reset()


@requires_openai_api_key
@pytest.mark.parametrize("wrapper", [TEMP_Wikipedia_DB], indirect=True)
def test_wikipedia_search(wrapper):
    results = list(wrapper.search("acinar cells of the salivary gland"))
    assert len(results) > 0
    for obj, _dist, _ in results:
        print(yaml.dump(obj))
    top_result = results[0][0]
    print(top_result)
    time.sleep(0.5)
    results2 = list(wrapper.search(top_result["title"]))
    assert len(results2) > 0
