import pytest
import yaml

from curate_gpt.agents.dae_agent import DatabaseAugmentedExtractor
from curate_gpt.extract.basic_extractor import BasicExtractor


@pytest.mark.parametrize(
    "query_term,query_property",
    [
        ("vacuole envelope", "label"),
        ("thylakoid membrane", "label"),
        ("A metabolic process that results in the breakdown of cysteine", "definition"),
    ],
)
def test_dalek(go_test_chroma_db, query_term, query_property):
    extractor = BasicExtractor()
    # extractor = RecursiveExtractor()
    # extractor = OpenAIExtractor()
    extractor.schema_proxy = go_test_chroma_db.schema_proxy
    dae = DatabaseAugmentedExtractor(knowledge_source=go_test_chroma_db, extractor=extractor)
    ao = dae.generate_extract(
        query_term, target_class="OntologyClass", context_property=query_property
    )
    print("RESULT:")
    print(yaml.dump(ao.object, sort_keys=False))
