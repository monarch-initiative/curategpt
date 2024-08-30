import logging
import os
import shutil
import tempfile
from pprint import pprint

import pytest
from oaklib import get_adapter
from oaklib.datamodels.obograph import GraphDocument

from curate_gpt import ChromaDBAdapter
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers.ontology.ontology_wrapper import OntologyWrapper
from tests import INPUT_DIR, OUTPUT_DIR
from tests.store.conftest import requires_openai_api_key

TEMP_OAKVIEW_DB = OUTPUT_DIR / "oaktmp"
TEMP_OAKVIEW_DB2 = OUTPUT_DIR / "oaktmp2"

# logger = logging.getLogger(__name__)

logging.basicConfig()
logger = logging.root
logger.setLevel(logging.DEBUG)


@pytest.fixture
def vstore():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db")
        adapter = get_adapter(INPUT_DIR / "go-nucleus.db")
        db = ChromaDBAdapter(db_path)
        wrapper = OntologyWrapper(oak_adapter=adapter, local_store=db, extractor=BasicExtractor())
        db.insert(wrapper.objects())
        yield wrapper


def test_oak_objects(vstore):
    """Test that the objects are extracted from the oak adapter."""
    shutil.rmtree(TEMP_OAKVIEW_DB, ignore_errors=True)
    # vstore.local_store.reset()
    objs = list(vstore.objects())
    [nucleus] = [obj for obj in objs if obj["id"] == "Nucleus"]
    assert nucleus["label"] == "nucleus"
    assert nucleus["original_id"] == "GO:0005634"
    reversed = vstore.unwrap_object(nucleus, store=vstore.local_store)
    nucleus = reversed.graphs[0].nodes[0]
    assert nucleus["lbl"] == "nucleus"
    assert nucleus["id"] == "GO:0005634"
    assert len(reversed.graphs[0].edges) == 2


def test_oak_index(vstore):
    """Test that the objects are indexed in the local store."""
    shutil.rmtree(TEMP_OAKVIEW_DB2, ignore_errors=True)
    adapter = get_adapter(INPUT_DIR / "go-nucleus.db")
    db = ChromaDBAdapter(str(TEMP_OAKVIEW_DB2))
    db.reset()
    wrapper = OntologyWrapper(oak_adapter=adapter, local_store=db, extractor=BasicExtractor())
    db.insert(wrapper.objects())
    g = wrapper.unwrap_object(
        {
            "id": "Nucleus",
            "label": "nucleus",
            "relationships": [{"predicate": "rdfs:subClassOf", "target": "Organelle"}],
            "original_id": "GO:0005634",
        },
        store=db,
    )
    if isinstance(g, GraphDocument):
        pprint(g.__dict__, width=100, indent=2)
        print(f"Number of Graphs in GraphDocument: {len(g.graphs)}")
        g = g.graphs[0] if g.graphs else None
        print("\n", g.nodes, "\n")
        print("\n", g.edges, "\n")
        # access node and edge attributes
        node = g.nodes[0]
        edge = g.edges[0]
        print(node.id, node.lbl)
        print(edge.sub, edge.pred, edge.obj)


@requires_openai_api_key
def test_oak_search(vstore):
    """Test that the objects are indexed and searchable in the local store."""
    results = list(vstore.search("nucl"))
    assert len(results) > 0
    assert any("nucleus" in result[0]["label"] for result in results)
