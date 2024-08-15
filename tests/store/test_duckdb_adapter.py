import itertools
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest
import yaml

from curate_gpt.store import CollectionMetadata
from curate_gpt.store.duckdb_adapter import DuckDBAdapter
from curate_gpt.store.schema_proxy import SchemaProxy
from curate_gpt.wrappers.ontology import OntologyWrapper
from linkml_runtime.utils.schema_builder import SchemaBuilder
from oaklib import get_adapter

from tests import INPUT_DBS, INPUT_DIR, OUTPUT_DIR, OUTPUT_DUCKDB_PATH

EMPTY_DB_PATH = os.path.join(OUTPUT_DIR, "empty_duckdb")


def terms_to_objects(terms: list[str]) -> list[Dict]:
    return [
        {
            "id": f"ID:{i}",
            "text": t,
            "wordlen": len(t),
            "nested": {"wordlen": len(t)},
            "vec": [float(i)] * 1536,
        }
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
    for i in db.list_collection_names():
        db.remove_collection(i)
    db.schema_proxy = simple_schema_manager
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
    results = list(db.search("fox", collection=collection, include=["metadatas"]))
    db.update(objs, collection=collection)
    assert db.collection_metadata(collection).description == "test collection"
    long_words = list(db.find(where={"wordlen": {"$gt": 12}}, collection=collection))
    assert len(long_words) == 2
    db.remove_collection(collection)
    db.insert(objs, collection=collection)
    results2 = list(db.search("fox", collection=collection, include=["metadatas"]))

    def _id(obj, dist, meta):
        return obj["id"]

    assert _id(*results[0]) == _id(*results2[0])
    results2 = list(db.find({}, limit=2, collection=collection))
    # because we cut the first metadata row (CollectionMetadata)
    assert len(results2) == 1



def test_the_embedding_function(simple_schema_manager, example_texts):
    db = DuckDBAdapter(OUTPUT_DUCKDB_PATH)
    db.conn.execute("DROP TABLE IF EXISTS test_collection")
    db.conn.execute("DROP TABLE IF EXISTS one_collection")
    db.conn.execute("DROP TABLE IF EXISTS test_ef_collection")
    db.conn.execute("DROP TABLE IF EXISTS test_openai_collection")
    db.conn.execute("DROP TABLE IF EXISTS test_openai_full_collection")
    objs = terms_to_objects(example_texts)
    db.insert(objs)
    assert db.collection_metadata(None).model == "all-MiniLM-L6-v2"
    assert db.collection_metadata(None).name == "test_collection"
    assert db.collection_metadata(None).hnsw_space == "cosine"
    db.insert(objs, collection="one_collection")
    assert db.collection_metadata("one_collection").model == "all-MiniLM-L6-v2"
    assert db.collection_metadata("one_collection").hnsw_space == "cosine"
    assert db.collection_metadata("one_collection").name == "one_collection"
    # db.insert(objs, collection="test_openai_collection", model="openai:")
    # print(db.collection_metadata("test_openai_collection").model)
    # assert db.collection_metadata("test_openai_collection").model == "openai:"
    # assert db.collection_metadata("test_openai_collection").hnsw_space == "cosine"
    # assert db.collection_metadata("test_openai_collection").name == "test_openai_collection"
    # db.insert(objs, collection="test_openai_full_collection", model="openai:text-embedding-ada-002")
    # assert db.collection_metadata("test_openai_full_collection").model == "openai:text-embedding-ada-002"
    # assert db.collection_metadata("test_openai_full_collection").hnsw_space == "cosine"
    # assert db.collection_metadata("test_openai_full_collection").name == "test_openai_full_collection"


@pytest.fixture
def ontology_db() -> DuckDBAdapter:
    db_path = os.path.join(INPUT_DBS, "go-nocleus-duck")
    db = DuckDBAdapter(db_path)
    # db.schema_proxy = SchemaProxy(ONTOLOGY_MODEL_PATH)
    [db.remove_collection(i) for i in db.list_collection_names()]
    return db


@pytest.fixture
def loaded_ontology_db(ontology_db) -> DuckDBAdapter:
    db = ontology_db
    for i in db.list_collection_names():
        db.remove_collection(i)
    adapter = get_adapter(str(INPUT_DIR / "go-nucleus.db"))
    view = OntologyWrapper(oak_adapter=adapter)
    ontology_db.text_lookup = view.text_field
    ontology_db.insert(view.objects(), collection="other_collection")
    ontology_db.text_lookup = "label"
    return db


def test_ontology_matches(ontology_db):
    collection = "test_collection"
    ontology_db.conn.execute("DROP TABLE IF EXISTS test_collection")
    adapter = get_adapter(str(INPUT_DIR / "go-nucleus.db"))
    view = OntologyWrapper(oak_adapter=adapter)
    ontology_db.text_lookup = view.text_field
    ontology_db.insert(view.objects(), collection=collection)
    ontology_db.text_lookup = "label"
    # TODO distances seem not right
    obj = ontology_db.lookup("Continuant", collection="test_collection")
    results = list(ontology_db.matches(obj, collection=collection))
    i = 0
    for obj, dist, meta in results:
        print(f"## {i} DISTANCE: {dist}")
        # print(f"## OBJECT: {obj}")
        print(f"META: {meta}")
        print(f"{yaml.dump(obj, sort_keys=False)}")
        i += 1

    assert len(results) == 10

    first_obj = results[0][0]
    print("the id", first_obj['id'])
    first_meta = results[0][2]
    new_id, new_definition = "Palm Beach", "A beach with palm trees"
    updated_obj = {
        "id": new_id,
        "label": first_obj["label"],
        "definition": new_definition,
        "aliases": first_obj["aliases"],
        "relationships": first_obj["relationships"],
        "logical_definition": first_obj["logical_definition"],
        "original_id": first_obj["original_id"],
    }

    # Update the object
    # Since we have control about indexing and delete an object before we update it in DuckDB
    # we can update an ID, as well as any other field in comparison to chromaDB
    ontology_db.update([updated_obj], collection=collection)
    # verify update
    updated_res = ontology_db.lookup(new_id, collection)
    assert updated_res['id'] == new_id
    assert updated_res['definition'] == new_definition
    assert updated_res['label'] == first_obj['label']

    # test upsert
    new_obj_insert = {
        "id": "Palm Beach",
        "key": "value"
    }
    ontology_db.upsert([new_obj_insert], collection="test_collection")
    # verify upsert
    new_results = ontology_db.lookup("Palm Beach", collection="test_collection")
    assert new_results["id"] == "Palm Beach"
    assert new_results['key'] == "value"


@pytest.mark.parametrize(
    "where,num_expected,limit,include",
    [
        ({"id": {"$eq": "Continuant"}}, 1, 10, ["metadata", "documents"]),
        ({"id": {"$eq": "NuclearMembrane"}}, 1, 10, ["metadata"]),
        ({"label": {"$eq": "continuant"}}, 1, 10, None),
    ],
)
def test_where_queries(loaded_ontology_db, where, num_expected, limit, include):
    db = loaded_ontology_db
    results = list(
        db.find(where=where, limit=limit, collection="other_collection", include=include)
    )
    assert len(results) == num_expected

@pytest.mark.parametrize(
    "batch_size",
    [
        100,
        500,
        1000,
        1500,
        2000,
        5000,
        10000,
        100000,
    ],
)
def test_load_in_batches(ontology_db, batch_size):
    adapter = get_adapter(str(INPUT_DIR / "go-nucleus.db"))
    view = OntologyWrapper(oak_adapter=adapter)
    ontology_db.text_lookup = view.text_field
    # start = time.time()
    ontology_db.insert(view.objects(), batch_size=batch_size, collection="other_collection")
    # end = time.time()
    # print(f"Time to insert {len(list(view.objects()))} objects with batch of {batch_size}: {end - start}")

    objs = list(
        ontology_db.find(collection="other_collection", limit=2000
        )
    )
    assert len(objs) > 100


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
        "pineapple helicopter 5",
        collection="test_collection",
        relevance_factor=relevance_factor,
        limit=20,
    )
    for obj, dist, _meta in results:
        print(f"{dist}\t{obj['text']}")


def test_diversified_search_on_empty_db(empty_db):
    relevance_factor = 0.5
    results = empty_db.search(
        "pineapple helicopter 5",
        collection="test_collection",
        relevance_factor=relevance_factor,
        limit=20,
    )
    assert len(list(results)) == 0
