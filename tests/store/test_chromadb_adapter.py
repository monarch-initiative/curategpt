from typing import Dict

import pytest
import yaml
from linkml_runtime.utils.schema_builder import SchemaBuilder
from oaklib import get_adapter

from curate_gpt import OntologyView
from curate_gpt.store.chromadb_adapter import ChromaDBAdapter
from curate_gpt.store.schema_manager import SchemaManager
from curate_gpt.view import ONTOLOGY_MODEL_PATH
from tests import INPUT_DBS, INPUT_DIR, OUTPUT_CHROMA_DB_PATH

texts = [
    "The quick brown fox jumps over the lazy dog",
    "canine",
    "vulpine",
    "let sleeping dogs lie",
    "chicken",
    "wings",
    "airplane",
]

combo_texts = [
    "pineapple helicopter 1",
    "pineapple helicopter 2",
    "apple helicopter",
    "guava airplane",
    "mango airplane",
    "papaya train",
    "banana train",
    "zucchini firetruck",
    "apple train",
    "orange lorry",
    "orange lorry chimney",
    "orange lorry window",
    "pineapple ship chimney",
    "cheese helicopter",
    "parmesan firefighter",
    "parmesan firefighter 5",
    "cheddar doctor",
    "swiss doctor",
    "helicopter apple",
    "chopper apple",
    "helicopter golden delicious",
]


def terms_to_objects(terms: list[str]) -> list[Dict]:
    return [{"id": f"ID:{i}", "text": t, "wordlen": len(t), "nested": {"wordlen": len(t)}} for i, t in enumerate(terms)]


@pytest.fixture
def simple_schema_manager() -> SchemaManager:
    sb = SchemaBuilder()
    sb.add_class("Term", slots=["id", "text", "wordlen", "nested"])
    sb.add_class("NestedObject", slots=["wordlen"])
    sb.add_slot("id", identifier=True, description="term id", replace_if_present=True)
    sb.add_slot("wordlen", range="integer", description="length of term", replace_if_present=True)
    sb.add_slot("nested", range="NestedObject", description="demonstrates nesting", replace_if_present=True)
    return SchemaManager(sb.schema)


def test_store(simple_schema_manager):
    db = ChromaDBAdapter(str(OUTPUT_CHROMA_DB_PATH))
    db.schema_manager = simple_schema_manager
    #db.linkml_schema_view = simple_schema
    db.client.reset()
    collection = "test"
    objs = terms_to_objects(texts)
    db.insert(objs, collection=collection)
    assert db.list_collection_names() == ["test"]
    results = list(db.search("fox", collection=collection))
    print(results)
    for obj in objs:
        print(f"QUERYING: {obj}")
        for match in db.matches(obj, collection=collection):
            print(f" - MATCH: {match}")
    db.update(objs, collection=collection)
    canines = list(db.find(where={"text": {"$eq": "canine"}}, collection=collection))
    print(canines)
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


@pytest.fixture
def ontology_db() -> ChromaDBAdapter:
    db = ChromaDBAdapter(str(INPUT_DBS / "go-nucleus-chroma"))
    db.schema_manager = SchemaManager(ONTOLOGY_MODEL_PATH)
    db.client.reset()
    return db


def test_ontology_matches(ontology_db):
    """
    Tests a pre-existing db
    """
    adapter = get_adapter(str(INPUT_DIR / "go-nucleus.db"))
    # ontology_db.linkml_schema_path = ONTOLOGY_MODEL_PATH
    view = OntologyView(adapter)
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


@pytest.fixture
def combo_db() -> ChromaDBAdapter:
    db = ChromaDBAdapter(str(OUTPUT_CHROMA_DB_PATH))
    db.client.reset()
    collection = "test"
    objs = terms_to_objects(combo_texts)
    db.insert(objs, collection=collection)
    return db

def test_diversified_search(combo_db):
    relevance_factor = 0.5
    results = combo_db.search("pineapple helicopter 5", collection="test", relevance_factor=relevance_factor, limit=20)
    for obj, dist, _meta in results:
        print(f"{dist}\t{obj['text']}")
