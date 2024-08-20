import logging
import shutil
from typing import Type

import pytest
import yaml

from curate_gpt import ChromaDBAdapter
from curate_gpt.agents.evidence_agent import EvidenceAgent
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers import BaseWrapper
from curate_gpt.wrappers.literature import PubmedWrapper, WikipediaWrapper
from tests import OUTPUT_DIR

TEMP_PUBMED_DB = OUTPUT_DIR / "pmid_tmp"

# logger = logging.getLogger(__name__)

logging.basicConfig()
logger = logging.root
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize(
    "source",
    [
        PubmedWrapper,
        WikipediaWrapper,
    ],
)
def test_evidence_inference(source: Type[BaseWrapper]):
    shutil.rmtree(TEMP_PUBMED_DB, ignore_errors=True)
    db = ChromaDBAdapter(str(TEMP_PUBMED_DB))
    extractor = BasicExtractor()
    db.reset()
    pubmed = source(local_store=db, extractor=extractor)
    ea = EvidenceAgent(chat_agent=pubmed)
    obj = {
        "label": "acinar cells of the salivary gland",
        "relationships": [
            {"predicate": "HasFunction", "object": "ManufactureSaliva"},
        ],
    }
    resp = ea.find_evidence(obj)
    print(yaml.dump(resp))
