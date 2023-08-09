from types import ModuleType
from typing import List

import pytest
from linkml_runtime.utils.schema_builder import SchemaBuilder
from pydantic import BaseModel

from curate_gpt.extract.basic_extractor import BasicExtractor
from curate_gpt.extract.extractor import AnnotatedObject, Extractor
from curate_gpt.extract.openai_extractor import OpenAIExtractor
from curate_gpt.extract.recursive_extractor import RecursiveExtractor
from curate_gpt.store.schema_manager import SchemaManager


class Occupation(BaseModel):
    category: str
    current: bool


class Person(BaseModel):
    name: str
    age: int
    occupations: List[Occupation]


@pytest.fixture
def schema_manager() -> SchemaManager:
    sb = SchemaBuilder("test")
    sb.add_class("Person", slots=["name", "age", "occupations"])
    sb.add_class("Occupation", slots=["category", "current"])
    sb.add_slot("age", range="integer", description="age in years", replace_if_present=True)
    sb.add_slot("occupations", range="Occupation", description="job held, and is it current", multivalued=True, replace_if_present=True)
    sb.add_slot("current", range="boolean", replace_if_present=True)
    sb.add_defaults()
    sm = SchemaManager(sb.schema)
    sm.pydantic_root_model = Person
    return sm


@pytest.mark.parametrize("extractor_type,kwargs,num_examples",
                         [
                              (RecursiveExtractor, {}, 0),
                              (RecursiveExtractor, {}, 99),
                              (OpenAIExtractor, {}, 99),
                              (OpenAIExtractor, {}, 0),
                              (OpenAIExtractor, {"examples_as_functions": True}, 99),
                              (BasicExtractor, {}, 99)]
                         )
def test_extract(extractor_type, kwargs, num_examples, schema_manager):
    extractor = extractor_type()
    extractor.schema_manager = schema_manager
    examples = [
        AnnotatedObject(
            object={"name": "John Doe", "age": 42, "occupations": [{"category": "Software Developer", "current": True}]},
            annotations={"text": "His name is John doe and he is 42 years old. He currently develops software for a living."},
        ),
        AnnotatedObject(
            object={"name": "Eleonore Li", "age": 27, "occupations": [{"category": "Physicist", "current": True}]},
            annotations={"text": "Eleonore Li is a 27 year old rising star Physicist."},
        ),
        AnnotatedObject(
            object={"name": "Lois Lane", "age": 24, "occupations": [{"category": "Reporter", "current": True}]},
            annotations={"text": "Lois Lane is a reporter for the daily planet. She is 24."},
        ),
        AnnotatedObject(
            object={"name": "Sandy Sands", "age": 33, "occupations": [{"category": "Costume Designer", "current": False}, {"category": "Architect", "current": True}]},
            annotations={"text": "the 33 year old Sandy Sands used to design costumes, now they are an architect."},
        ),
    ]
    successes = []
    failures = []
    for i in range(0, len(examples)):
        print(f"ITERATION {i} // {extractor_type}")
        test = examples[i]
        train = examples[:i] + examples[i + 1:]
        result = extractor.extract(target_class="Person", examples=train[0:num_examples], text=test.text, **kwargs)
        print(f"RESULTS:")
        print(result)
        if result.object == test.object:
            print("SUCCESS")
            successes.append(result)
        else:
            print(f"FAILURE: expected={test.object}")
            failures.append(result)
    print(f"{extractor_type} {kwargs} {num_examples} SUCCESSES: {len(successes)} FAILURES: {len(failures)}")