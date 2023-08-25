import pytest
import yaml

from curate_gpt.agents.dac_agent import DatabaseAugmentedCompletion
from curate_gpt.agents.dase_agent import DatabaseAugmentedStructuredExtraction
from curate_gpt.extract.basic_extractor import BasicExtractor


@pytest.mark.parametrize(
    "text",
    [
        "A metabolic process that results in the breakdown of cysteine",
    ],
)
def test_object_completion(go_test_chroma_db, text):
    extractor = BasicExtractor()
    # extractor = RecursiveExtractor()
    # extractor = OpenAIExtractor()
    extractor.schema_proxy = go_test_chroma_db.schema_proxy
    dase = DatabaseAugmentedStructuredExtraction(
        knowledge_source=go_test_chroma_db, extractor=extractor
    )
    ao = dase.extract(
        text,
        target_class="OntologyClass",
    )
    print("RESULT:")
    print(yaml.dump(ao.object, sort_keys=False))


def test_predict_missing(go_test_chroma_db):
    extractor = BasicExtractor()
    dae = DatabaseAugmentedCompletion(knowledge_source=go_test_chroma_db, extractor=extractor)
    n = 0
    for row in dae.generate_all(collection="test", field_to_predict="definition"):
        n += 1
        print(row)
        if n > 4:
            break


def test_de_novo(go_test_chroma_db):
    extractor = BasicExtractor()
    dae = DatabaseAugmentedCompletion(knowledge_source=go_test_chroma_db, extractor=extractor)
    ao = dae.complete(
        {},
        target_class="OntologyClass",
        rules=[
            "create a term for the envelope that surrounds a thylakoid (a membrane-bounded compartment)"
        ],
    )
    print("RESULT:")
    print(yaml.dump(ao.object, sort_keys=False))
