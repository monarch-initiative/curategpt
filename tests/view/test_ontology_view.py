from oaklib import get_adapter
from oaklib.interfaces import OboGraphInterface

from curate_gpt.store.chromadb_adapter import ChromaDBAdapter
from curate_gpt.view.ontology_view import OntologyView
from tests import INPUT_DIR, OUTPUT_CHROMA_DB_PATH


def test_view():
    oak_adapter = get_adapter(INPUT_DIR / "go-nucleus.db")
    view = OntologyView(oak_adapter)
    view.as_object("GO:0005634")

    db = ChromaDBAdapter(str(OUTPUT_CHROMA_DB_PATH))
    db.text_lookup = view.text_field
    db.reset()
    db.insert(view.objects())
    for curie in oak_adapter.entities():
        obj = view.as_object(curie)
        if not obj:
            continue
        print(f"QUERYING: {obj}")
        for m in db.matches(obj):
            print(f" - MATCH: {m}")
