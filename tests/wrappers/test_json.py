import logging
import shutil

import pytest
import yaml
from oaklib import get_adapter

from curate_gpt import ChromaDBAdapter
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers import get_wrapper
from curate_gpt.wrappers.ontology.ontology_wrapper import OntologyWrapper
from tests import INPUT_DIR, OUTPUT_DIR


def test_json_objects():
    wrapper = get_wrapper("json")
    wrapper.source_locator = INPUT_DIR / "biolink-model.yaml"
    wrapper.path_expression = "$.classes[*]"
    objs = list(wrapper.objects())
    assert len(objs) == 1
    assert "gene" in objs[0]
