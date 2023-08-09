import pytest
import yaml

from curate_gpt import OntologyView
from curate_gpt.extract.basic_extractor import BasicExtractor
from curate_gpt.extract.openai_extractor import OpenAIExtractor
from curate_gpt.extract.recursive_extractor import RecursiveExtractor
from curate_gpt.agents.rage import RetrievalAugmentedExtractor
from curate_gpt.store.chromadb_adapter import ChromaDBAdapter
from curate_gpt.store.schema_manager import SchemaManager
from curate_gpt.view import ONTOLOGY_MODEL_PATH
from curate_gpt.view.ontology_view import Ontology, OntologyClass
from tests import INPUT_DBS


@pytest.fixture
def chroma_db() -> ChromaDBAdapter:
    db = ChromaDBAdapter(str(INPUT_DBS / "go-nucleus-chroma"))
    db.schema_manager = SchemaManager(ONTOLOGY_MODEL_PATH)
    m = OntologyClass
    return db

@pytest.mark.parametrize("query_term,query_property",
    [
        ("vacuole envelope", "label"),
        ("thylakoid membrane", "label"),
        ("A metabolic process that results in the breakdown of cysteine", "definition"),
    ])
def test_rage(chroma_db,query_term, query_property):
    #extractor = BasicExtractor()
    extractor = RecursiveExtractor()
    extractor = OpenAIExtractor()
    extractor.schema_manager = chroma_db.schema_manager
    rage = RetrievalAugmentedExtractor(kb_adapter=chroma_db, extractor=extractor)
    ao = rage.generate_extract(query_term, target_class="OntologyClass", context_property=query_property)
    print("RESULT:")
    print(yaml.dump(ao.object, sort_keys=False))
