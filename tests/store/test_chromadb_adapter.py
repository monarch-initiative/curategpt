import shutil
from typing import Dict

import pytest
import yaml
from linkml_runtime.utils.schema_builder import SchemaBuilder
from oaklib import get_adapter

from curate_gpt.store.chromadb_adapter import ChromaDBAdapter
from curate_gpt.store.schema_proxy import SchemaProxy
from curate_gpt.wrappers.ontology import ONTOLOGY_MODEL_PATH, OntologyWrapper
from tests import INPUT_DBS, INPUT_DIR, OUTPUT_CHROMA_DB_PATH, OUTPUT_DIR

EMPTY_DB_PATH = OUTPUT_DIR / "empty_db"


def terms_to_objects(terms: list[str]) -> list[Dict]:
    return [
        {"id": f"ID:{i}", "text": t, "wordlen": len(t), "nested": {"wordlen": len(t)}}
        for i, t in enumerate(terms)
    ]


@pytest.fixture
def empty_db() -> ChromaDBAdapter:
    shutil.rmtree(EMPTY_DB_PATH, ignore_errors=True)
    db = ChromaDBAdapter(str(EMPTY_DB_PATH))
    collection = "test"
    objs = []
    db.insert(objs, collection=collection)
    return db


@pytest.fixture
def simple_schema_manager() -> SchemaProxy:
    sb = SchemaBuilder()
    sb.add_class("Term", slots=["id", "text", "wordlen", "nested"])
    sb.add_class("NestedObject", slots=["wordlen"])
    sb.add_slot("id", identifier=True, description="term id", replace_if_present=True)
    sb.add_slot("wordlen", range="integer", description="length of term", replace_if_present=True)
    sb.add_slot(
        "nested", range="NestedObject", description="demonstrates nesting", replace_if_present=True
    )
    return SchemaProxy(sb.schema)


def test_store(simple_schema_manager, example_texts):
    db = ChromaDBAdapter(str(OUTPUT_CHROMA_DB_PATH))
    db.schema_proxy = simple_schema_manager
    db.client.reset()
    assert db.list_collection_names() == []
    collection = "test"
    objs = terms_to_objects(example_texts)
    db.insert(objs, collection=collection)
    md = db.collection_metadata(collection)
    md.description = "test collection"
    db.set_collection_metadata(collection, md)
    assert db.collection_metadata(collection).description == "test collection"
    db2 = ChromaDBAdapter(str(OUTPUT_CHROMA_DB_PATH))
    assert db2.collection_metadata(collection).description == "test collection"
    assert db.list_collection_names() == ["test"]
    results = list(db.search("fox", collection=collection))
    print(results)
    for obj in objs:
        print(f"QUERYING: {obj}")
        for match in db.matches(obj, collection=collection):
            print(f" - MATCH: {match}")
    db.update(objs, collection=collection)
    assert db.collection_metadata(collection).description == "test collection"
    canines = list(db.find(where={"text": {"$eq": "canine"}}, collection=collection))
    print(f"CANINES: {canines}")
    long_words = list(db.find(where={"wordlen": {"$gt": 12}}, collection=collection))
    print(long_words)
    assert len(long_words) == 2
    db.remove_collection(collection)
    db.insert(objs, collection=collection)
    results2 = list(db.search("fox", collection=collection))

    def _id(obj, _dist, _meta):
        return obj["id"]

    assert _id(*results[0]) == _id(*results2[0])
    limit = 5
    results2 = list(db.find({}, limit=5, collection=collection))
    assert len(results2) == limit
    results2 = list(db.find({}, limit=10000000, collection=collection))
    assert len(results2) > limit


def test_autoschema(example_texts):
    db = ChromaDBAdapter(str(OUTPUT_CHROMA_DB_PATH))
    db.client.reset()
    collection = "auto"
    objs = terms_to_objects(example_texts)
    db.insert(objs, collection=collection)
    print(db.schema_proxy)
    fields = db.field_names(collection=collection)
    assert "text" in fields
    fields2 = db.field_names(collection=collection)
    assert fields == fields2


def test_embedding_function(simple_schema_manager, example_texts):
    """
    Tests having two collections with different models and embedding functions.

    :param simple_schema_manager:
    :return:
    """
    db = ChromaDBAdapter(str(OUTPUT_CHROMA_DB_PATH))
    db.reset()
    objs = terms_to_objects(example_texts)
    db.insert(objs[1:])
    db.insert(objs[1:], collection="default_ef", model=None)
    db.insert(objs[1:], collection="openai", model="openai:")
    assert db.collection_metadata("default_ef").name == "default_ef"
    assert db.collection_metadata("openai").name == "openai"
    assert db.collection_metadata(None).model == "all-MiniLM-L6-v2"
    assert db.collection_metadata("default_ef").model == "all-MiniLM-L6-v2"
    assert db.collection_metadata("openai").model == "openai:"
    db.insert([objs[0]])
    db.insert([objs[0]], collection="default_ef")
    db.insert([objs[0]], collection="openai")
    assert db.collection_metadata("default_ef").model == "all-MiniLM-L6-v2"
    assert db.collection_metadata("openai").model == "openai:"
    results_ef = list(db.search("fox", collection="default_ef"))
    results_oai = list(db.search("fox", collection="openai"))
    assert len(results_ef) > 0
    assert len(results_oai) > 0
    with pytest.raises(ValueError):
        db.update_collection_metadata("default_ef", model="openai:")
    with pytest.raises(ValueError):
        db.update_collection_metadata("openai", model="all-MiniLM-L6-v2")


@pytest.fixture
def ontology_db() -> ChromaDBAdapter:
    db = ChromaDBAdapter(str(INPUT_DBS / "go-nucleus-chroma"))
    db.schema_proxy = SchemaProxy(ONTOLOGY_MODEL_PATH)
    db.client.reset()
    # db.default_model = "openai:"
    return db


@pytest.fixture
def loaded_ontology_db(ontology_db) -> ChromaDBAdapter:
    db = ontology_db
    adapter = get_adapter(str(INPUT_DIR / "go-nucleus.db"))
    # ontology_db.linkml_schema_path = ONTOLOGY_MODEL_PATH
    view = OntologyWrapper(oak_adapter=adapter)
    ontology_db.text_lookup = view.text_field
    ontology_db.insert(view.objects(), collection="terms_go")
    ontology_db.text_lookup = "label"
    return db


def test_ontology_matches(ontology_db):
    """
    Tests a pre-existing db
    """
    adapter = get_adapter(str(INPUT_DIR / "go-nucleus.db"))
    # ontology_db.linkml_schema_path = ONTOLOGY_MODEL_PATH
    view = OntologyWrapper(oak_adapter=adapter)
    ontology_db.text_lookup = view.text_field
    ontology_db.insert(view.objects())
    # TODO
    ontology_db.text_lookup = "label"
    obj = ontology_db.lookup("NuclearMembrane")
    results = ontology_db.matches(obj)
    i = 0
    for obj, distance, _meta in results:
        print(f"## {i} DISTANCE: {distance}")
        print(yaml.dump(obj, sort_keys=False))


@pytest.mark.parametrize(
    "where,num_expected,limit",
    [
        ({"id": {"$eq": "NuclearMembrane"}}, 1, None),
        ({"id": {"$eq": "NuclearMembrane"}}, 1, 100000),
        # ({"aliases": {"$eq": None}}, 1, None),
        ({}, 10, 10),
        ({}, 1, 1),
        ({}, 280, None),
    ],
)
def test_where_queries(loaded_ontology_db, where, num_expected, limit):
    """
    Tests use of where clauses
    """
    db = loaded_ontology_db
    results = list(db.find(where=where, limit=limit, collection="terms_go"))
    # for r in results:
    #    logger.debug(r)
    assert len(results) == num_expected


def test_load_in_batches(ontology_db):
    """
    Tests ability to load in batches
    """
    adapter = get_adapter(str(INPUT_DIR / "go-nucleus.db"))
    # ontology_db.linkml_schema_path = ONTOLOGY_MODEL_PATH
    view = OntologyWrapper(oak_adapter=adapter)
    ontology_db.text_lookup = view.text_field
    ontology_db.insert(view.objects(), batch_size=10, collection="test")
    objs = list(ontology_db.find(collection="test", limit=2000))
    assert len(objs) > 100


@pytest.fixture
def combo_db(example_combo_texts) -> ChromaDBAdapter:
    db = ChromaDBAdapter(str(OUTPUT_CHROMA_DB_PATH))
    db.client.reset()
    collection = "test"
    objs = terms_to_objects(example_combo_texts)
    db.insert(objs, collection=collection)
    return db


def test_diversified_search(combo_db):
    relevance_factor = 0.5
    results = combo_db.search(
        "pineapple helicopter 5", collection="test", relevance_factor=relevance_factor, limit=20
    )
    for obj, dist, _meta in results:
        print(f"{dist}\t{obj['text']}")


def test_diversified_search_on_empty_db(empty_db):
    relevance_factor = 0.5
    results = empty_db.search(
        "pineapple helicopter 5", collection="test", relevance_factor=relevance_factor, limit=20
    )
    assert len(list(results)) == 0
