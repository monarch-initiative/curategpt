import logging
from pathlib import Path
from pprint import pprint

import pytest
from oaklib import get_adapter
from oaklib.datamodels.obograph import GraphDocument
from venomx.model.venomx import Dataset, Index, Model, ModelInputMethod

from curategpt.extract import BasicExtractor
from curategpt.wrappers.ontology.ontology_wrapper import OntologyWrapper
from tests import INPUT_DIR, OUTPUT_DIR
from tests.store.conftest import requires_openai_api_key
from tests.utils.helper import DEBUG_MODE, create_db_dir, setup_db

TEMP_OAK_OBJ = OUTPUT_DIR / "oak_tmp_obj"
TEMP_OAK_IND = OUTPUT_DIR / "oak_tmp_ind"
TEMP_OAK_SEARCH = OUTPUT_DIR / "oak_tmp_search"

# logger = logging.getLogger(__name__)

logging.basicConfig()
logger = logging.root
logger.setLevel(logging.DEBUG)

# for debugging meanwhile implementing
def test_insert_without_venomx():
    db = setup_db(Path("../db"))
    collection_name = "test_collection_without_venomx_set_upfront"

    extractor = BasicExtractor()
    adapter = get_adapter(INPUT_DIR / "go-nucleus.db")
    wrapper = OntologyWrapper(oak_adapter=adapter, local_store=db, extractor=extractor)

    # venomx = Index(
    #     id=f"{collection_name}",
    #     dataset=Dataset(
    #         name="test_ont_hp"
    #     ),
    #     embedding_model=Model(
    #         name=db.default_model
    #     ),
    #     embedding_input_method=ModelInputMethod(
    #         fields=['label']
    #     )
    # )
    #
    # print(venomx)


    db.insert(
        wrapper.objects(),
        collection=collection_name,
        # venomx=venomx
    )
    names = db.list_collection_names()
    print(names)

    col = db.client.get_collection(f"{collection_name}")
    metadata = col.metadata
    print(metadata)

# for debugging meanwhile implementing
def test_insert_with_venomx():
    db = setup_db(Path("../db"))
    collection_name = "test_collection_with_venomx_set_upfront"

    extractor = BasicExtractor()
    adapter = get_adapter(INPUT_DIR / "go-nucleus.db")
    wrapper = OntologyWrapper(oak_adapter=adapter, local_store=db, extractor=extractor)

    venomx = Index(
        id=f"{collection_name}",
        dataset=Dataset(
            name="test_ont_hp"
        ),
        embedding_model=Model(
            name=db.default_model
        ),
        embedding_input_method=ModelInputMethod(
            fields=['label']
        )
    )

    # print(venomx)
    # print("\n\n", venomx.id ,"\n\n")

    db.insert(
        wrapper.objects(),
        collection=collection_name,
        venomx=venomx
    )
    names = db.list_collection_names()
    print(names)
    col = db.client.get_collection(f"{collection_name}")
    metadata = col.metadata
    print(metadata)

@pytest.fixture
def vstore(request, tmp_path):
    temp_db_base = request.param
    temp_dir = create_db_dir(tmp_path, temp_db_base)
    db = setup_db(temp_dir)
    extractor = BasicExtractor()
    # mock, possible connection error?
    adapter = get_adapter(INPUT_DIR / "go-nucleus.db")
    try:
        wrapper = OntologyWrapper(oak_adapter=adapter, local_store=db, extractor=extractor)
        db.insert(wrapper.objects())
        yield wrapper, db
    except Exception as e:
        raise e
    finally:
        if not DEBUG_MODE:
            db.reset()


@pytest.mark.parametrize('vstore', [TEMP_OAK_OBJ], indirect=True)
def test_oak_objects(vstore):
    """Test that the objects are extracted from the oak adapter."""
    wrapper, _ = vstore
    objs = list(wrapper.objects())
    [nucleus] = [obj for obj in objs if obj["id"] == "Nucleus"]
    assert nucleus["label"] == "nucleus"
    assert nucleus["original_id"] == "GO:0005634"
    reversed = wrapper.unwrap_object(nucleus, store=wrapper.local_store)
    nucleus = reversed.graphs[0].nodes[0]
    assert nucleus["lbl"] == "nucleus"
    assert nucleus["id"] == "GO:0005634"
    assert len(reversed.graphs[0].edges) == 2


@pytest.mark.parametrize('vstore', [TEMP_OAK_IND], indirect=True)
def test_oak_index(vstore):
    """Test that the objects are indexed in the local store."""
    wrapper, _ = vstore
    g = wrapper.unwrap_object(
        {
            "id": "Nucleus",
            "label": "nucleus",
            "relationships": [{"predicate": "rdfs:subClassOf", "target": "Organelle"}],
            "original_id": "GO:0005634",
        },
        store=wrapper.local_store,
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


@pytest.mark.parametrize('vstore', [TEMP_OAK_SEARCH], indirect=True)
@requires_openai_api_key
def test_oak_search(vstore):
    """Test that the objects are indexed and searchable in the local store."""
    _, db = vstore
    results = list(db.search("nucl"))
    assert len(results) > 0
    assert any("nucleus" in result[0]["label"] for result in results)
