from typing import Dict, List

import pytest
from linkml_runtime.utils.schema_builder import SchemaBuilder

from curate_gpt import DBAdapter
from curate_gpt.store import get_store
from curate_gpt.store.schema_proxy import SchemaProxy
from tests import OUTPUT_DIR

EMPTY_DB_PATH = OUTPUT_DIR / "empty_db"

# TODO: DRY


def terms_to_objects(terms: list[str]) -> list[Dict]:
    return [
        {"id": f"ID:{i}", "text": t, "wordlen": len(t), "nested": {"wordlen": len(t)}}
        for i, t in enumerate(terms)
    ]


@pytest.fixture
def empty_db() -> DBAdapter:
    store = get_store("in_memory")
    return store


# TODO: move to conftest.py
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
    db = get_store("in_memory")
    db.schema_proxy = simple_schema_manager
    assert db.list_collection_names() == []
    collection = "test"
    objs = terms_to_objects(example_texts)
    db.insert(objs, collection=collection)
    md = db.collection_metadata(collection)
    md.description = "test collection"
    db.set_collection_metadata(collection, md)
    assert db.collection_metadata(collection).description == "test collection"
