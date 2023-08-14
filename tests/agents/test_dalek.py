import pytest
import yaml

from curate_gpt import OntologyView
from curate_gpt.agents.dalek import DatabaseAugmentedExtractor
from curate_gpt.extract.basic_extractor import BasicExtractor
from curate_gpt.extract.openai_extractor import OpenAIExtractor
from curate_gpt.extract.recursive_extractor import RecursiveExtractor
from curate_gpt.store.chromadb_adapter import ChromaDBAdapter
from curate_gpt.store.schema_proxy import SchemaProxy
from curate_gpt.view import ONTOLOGY_MODEL_PATH
from curate_gpt.view.ontology_view import Ontology, OntologyClass
from tests import INPUT_DBS


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
    dae = DatabaseAugmentedExtractor(kb_adapter=go_test_chroma_db, extractor=extractor)
    ao = dae.generate_extract(
        query_term, target_class="OntologyClass", context_property=query_property
    )
    print("RESULT:")
    print(yaml.dump(ao.object, sort_keys=False))
