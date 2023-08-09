import pytest
import yaml

from curate_gpt import OntologyView
from curate_gpt.rag.openai_rag import OpenAIRAG
from curate_gpt.store.chromadb_adapter import ChromaDBAdapter
from curate_gpt.view.ontology_view import Ontology
from tests import INPUT_DBS


def test_rag():
    db = ChromaDBAdapter(str(INPUT_DBS / "go-nucleus-chroma"))
    rag = OpenAIRAG(db_adapter=db, root_class=Ontology)
    obj = rag.generate("create terms representing a vacuole envelope")
    print("RESULT:")
    print(yaml.dump(obj, sort_keys=False))


@pytest.mark.skip("Too slow")
def test_conversation():
    db = ChromaDBAdapter(str(INPUT_DBS / "go-nucleus-chroma"))
    rag = OpenAIRAG(db_adapter=db, root_class=Ontology, conversation_mode=True)
    obj = rag.generate("create a term representing a vacuole envelope")
    print("RESULT:")
    print(yaml.dump(obj, sort_keys=False))
    obj = rag.generate("generate the same term again, with more concise definitions")
    print("RESULT 2:")
    print(yaml.dump(obj, sort_keys=False))
