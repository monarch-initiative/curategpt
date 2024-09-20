import logging
from typing import Type

import pytest
import yaml

from curate_gpt.agents.evidence_agent import EvidenceAgent
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers import BaseWrapper
from curate_gpt.wrappers.literature import PubmedWrapper, WikipediaWrapper

from tests import OUTPUT_DIR
from tests.store.conftest import requires_openai_api_key
from tests.utils.helper import create_db_dir, setup_db, DEBUG_MODE

TEMP_PUBMED_DB = OUTPUT_DIR / "pmid_tmp"

# logger = logging.getLogger(__name__)

logging.basicConfig()
logger = logging.root
logger.setLevel(logging.DEBUG)


@requires_openai_api_key
@pytest.mark.parametrize(
    "source",
    [
        PubmedWrapper,
        WikipediaWrapper,
    ],
)
def test_evidence_inference(tmp_path, source: Type[BaseWrapper]):
    tmp_dir = create_db_dir(tmp_path=tmp_path, out_dir=TEMP_PUBMED_DB)
    db = setup_db(tmp_dir)
    extractor = BasicExtractor()
    try:
        pubmed = source(local_store=db, extractor=extractor)
    except Exception as e:
        raise e
    finally:
        if not DEBUG_MODE:
            if tmp_dir.exists():
                db.reset()

    ea = EvidenceAgent(chat_agent=pubmed)
    obj = {
        "label": "acinar cells of the salivary gland",
        "relationships": [
            {"predicate": "HasFunction", "object": "ManufactureSaliva"},
        ],
    }
    resp = ea.find_evidence(obj)
    print(yaml.dump(resp))
