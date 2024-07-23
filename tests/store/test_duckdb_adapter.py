import itertools
import os
import shutil
from typing import Dict

import pytest

from curate_gpt.store import CollectionMetadata
from curate_gpt.store.duckdb_adapter import DuckDBAdapter
from curate_gpt.store.schema_proxy import SchemaProxy
from curate_gpt.wrappers.ontology import ONTOLOGY_MODEL_PATH, OntologyWrapper
from linkml_runtime.utils.schema_builder import SchemaBuilder
from oaklib import get_adapter

from tests import INPUT_DBS, INPUT_DIR, OUTPUT_DUCKDB_PATH, OUTPUT_DIR

EMPTY_DB_PATH = os.path.join(OUTPUT_DIR, "empty_duckdb")


def terms_to_objects(terms: list[str]) -> list[Dict]:
    return [
        {"id": f"ID:{i}", "text": t, "wordlen": len(t), "nested": {"wordlen": len(t)}, "vec": [float(i)] * 1536}
        for i, t in enumerate(terms)
    ]


@pytest.fixture
def empty_db() -> DuckDBAdapter:
    shutil.rmtree(EMPTY_DB_PATH, ignore_errors=True)
    db = DuckDBAdapter(EMPTY_DB_PATH)
    db.conn.execute("DROP TABLE IF EXISTS collection_metadata")
    db.conn.execute("DROP TABLE IF EXISTS test_collection")
    collection = "test_collection"
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
    db = DuckDBAdapter(OUTPUT_DUCKDB_PATH)
    db.schema_proxy = simple_schema_manager
    db.conn.execute("DROP TABLE IF EXISTS test_collection")
    db.conn.execute("DROP TABLE IF EXISTS test_ef_collection")
    db.conn.execute("DROP TABLE IF EXISTS test_openai_collection")
    assert db.list_collection_names() == []
    collection = "test_collection"
    objs = terms_to_objects(example_texts)
    db.insert(objs, collection=collection)
    md = db.collection_metadata(collection)
    md.description = "test collection"
    db.set_collection_metadata(collection, md)
    assert db.collection_metadata(collection).description == "test collection"
    db2 = DuckDBAdapter(str(OUTPUT_DUCKDB_PATH))
    assert db2.collection_metadata(collection).description == "test collection"
    assert db.list_collection_names() == ["test_collection"]
    results = list(db.search("fox", collection=collection, include=['metadata']))
    db.update(objs, collection=collection)
    assert db.collection_metadata(collection).description == "test collection"
    long_words = list(db.find(where={"wordlen": {"$gt": 12}}, collection=collection))
    assert len(long_words) == 2
    db.remove_collection(collection)
    db.insert(objs, collection=collection)
    results2 = list(db.search("fox", collection=collection, include=['metadata']))

    def _id(id, _meta, _emb, _doc, _dist):
        return id

    assert _id(*results[0]) == _id(*results2[0])
    limit = 2
    results2 = list(db.find({}, limit=2, collection=collection))
    assert len(results2) == limit
    results2 = list(db.find({}, limit=10000000, collection=collection))
    assert len(results2) > limit

def test_the_embedding_function(simple_schema_manager, example_texts):
    db = DuckDBAdapter(OUTPUT_DUCKDB_PATH)
    db.conn.execute("DROP TABLE IF EXISTS test_collection")
    db.conn.execute("DROP TABLE IF EXISTS test_ef_collection")
    db.conn.execute("DROP TABLE IF EXISTS test_openai_collection")
    objs = terms_to_objects(example_texts)
    db.insert(objs[1:], collection="test_collection")
    metadata_default = db.collection_metadata("test_collection")
    if metadata_default is None:
        md_default = CollectionMetadata(name="test_collection", model=db.default_model)
        assert md_default.model == "all-MiniLM-L6-v2"
        db.set_collection_metadata("test_collection", md_default)
        metadata_default = db.collection_metadata("test_collection")
    db.insert(objs[1:], collection="test_ef_collection", model=None)
    md_ef = CollectionMetadata(name="test_ef_collection", model="openai:")
    db.set_collection_metadata("test_ef_collection", md_ef)
    assert md_ef.model == "openai:"
    # test openai model
    db.insert(objs[1:], collection="test_openai_collection", model="openai:text-embedding-ada-002")


@pytest.fixture
def ontology_db() -> DuckDBAdapter:
    db_path = os.path.join(INPUT_DBS, "go-nocleus-duck")
    db = DuckDBAdapter(db_path)
    # db.schema_proxy = SchemaProxy(ONTOLOGY_MODEL_PATH)
    db.conn.execute("DROP TABLE IF EXISTS test_collection")
    db.conn.execute("DROP TABLE IF EXISTS other_collection")
    db.conn.execute("DROP TABLE IF EXISTS terms_go_collection")
    return db


@pytest.fixture
def loaded_ontology_db(ontology_db) -> DuckDBAdapter:
    db = ontology_db
    list_all = db.list_collection_names()
    db.conn.execute("DROP TABLE IF EXISTS other_collection")
    db.conn.execute("DROP TABLE IF EXISTS test_collection")
    db.conn.execute("DROP TABLE IF EXISTS terms_go_collection")
    print(f"LIST ALL: {list_all}")
    adapter = get_adapter(str(INPUT_DIR / "go-nucleus.db"))
    view = OntologyWrapper(oak_adapter=adapter)
    ontology_db.text_lookup = view.text_field
    sliced_gen = list(itertools.islice(view.objects(), 3))
    ontology_db.insert(sliced_gen, collection="other_collection")
    ontology_db.text_lookup = "label"
    return db


def test_ontology_matches(ontology_db):
    ontology_db.conn.execute("DROP TABLE IF EXISTS test_collection")
    adapter = get_adapter(str(INPUT_DIR / "go-nucleus.db"))
    view = OntologyWrapper(oak_adapter=adapter)
    ontology_db.text_lookup = view.text_field
    ontology_db.insert(view.objects())
    ontology_db.text_lookup = "label"
    obj = ontology_db.lookup("Continuant", collection="test_collection")
    results = list(ontology_db.matches(obj))
    assert len(list(results)) == 10
    # test update
    first_obj = results[0]
    updated_obj = {
        "id": first_obj.id,
        "metadata": {"updated_key": "updated_value"},
        "embeddings": [0.1] * len(first_obj.embeddings),
        "documents": "Updated document text"
    }
    ontology_db.update([updated_obj], collection="test_collection")
    # verify update
    result = ontology_db.lookup(first_obj.id, collection="test_collection")
    assert result['metadata'] == {"updated_key": "updated_value"}
    assert result == updated_obj
    updated_results = list(ontology_db.matches(updated_obj))
    assert len(updated_results) == 10
    for res in updated_results:
        if res.id == first_obj.id:
            assert res.metadata == updated_obj
            assert res.metadata['metadata'] == {"updated_key": "updated_value"}
    # test upsert
    new_obj = {
        "id": "new_id",
        "metadata": {"new_key": "new_value"},
        "embeddings": [0.5] * len(first_obj.embeddings),
        "documents": "New document text"
    }
    ontology_db.upsert([new_obj], collection="test_collection")
    # verify upsert
    new_results = ontology_db.lookup("new_id", collection="test_collection")
    assert new_results["metadata"] == {"new_key": "new_value"}


@pytest.mark.parametrize(
    "where,num_expected,limit,include",
    [
        ({"id": {"$eq": "Continuant"}}, 1, 10, ["id", "metadata", "embeddings", "documents"]),
        ({"id": {"$eq": "Continuant"}}, 1, 10, ["id", "metadata"]),
        ({"label": {"$eq": "continuant"}}, 1, 10, None),
    ],
)
def test_where_queries(loaded_ontology_db, where, num_expected, limit, include):
    db = loaded_ontology_db
    results = list(db.find(where=where, limit=limit, collection="other_collection", include=include))
    assert len(results) == num_expected
    for res in results:
        if include:
            if 'id' in include:
                assert res.id is not None
            if 'metadata' in include:
                assert res.metadata is not None
            if 'embeddings' in include:
                assert res.embeddings is not None
            if 'documents' in include:
                assert res.documents is not None
            if 'distance' in include:
                assert res.distance is not None
        else:
            assert res.id is not None
            assert res.metadata is not None
            assert res.embeddings is not None
            assert res.documents is not None


def test_load_in_batches(ontology_db):
    adapter = get_adapter(str(INPUT_DIR / "go-nucleus.db"))
    view = OntologyWrapper(oak_adapter=adapter)
    ontology_db.text_lookup = view.text_field
    sliced_gen = list(itertools.islice(view.objects(), 3))
    ontology_db.insert(sliced_gen, batch_size=10, collection="other_collection")
    objs = list(
        ontology_db.find(where={"original_id": {"$eq": "BFO:0000002"}}, collection="other_collection", limit=2000))
    assert len(objs) == 1


@pytest.fixture
def combo_db(example_combo_texts) -> DuckDBAdapter:
    db = DuckDBAdapter(str(OUTPUT_DUCKDB_PATH))
    db.conn.execute("DROP TABLE IF EXISTS test_collection")
    collection = "test_collection"
    objs = terms_to_objects(example_combo_texts)
    db.insert(objs, collection=collection)
    return db


def test_diversified_search(combo_db):
    relevance_factor = 0.5
    results = combo_db.search(
        "pineapple helicopter 5", collection="test_collection", relevance_factor=relevance_factor, limit=20
    )
    for i, res in enumerate(results):
        obj, distance, id_field, doc = res.metadata, res.distance, res.id, res.documents
        print(f"## {i} DISTANCE: {distance}")
        print(f"ID: {id_field}")
        print(f"DOC: {doc}")
        if i >= 2:
            break


def test_diversified_search_on_empty_db(empty_db):
    relevance_factor = 0.5
    results = empty_db.search(
        "pineapple helicopter 5", collection="test_collection", relevance_factor=relevance_factor, limit=20
    )
    assert len(list(results)) == 0
