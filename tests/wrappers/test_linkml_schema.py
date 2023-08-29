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


def test_linkml_schema_objects():
    wrapper = get_wrapper("linkml_schema")
    wrapper.source_locator = INPUT_DIR / "biolink-model.yaml"
    objs = list(wrapper.objects())
    genes = [o for o in objs if o["name"] == "gene"]
    assert len(genes) == 1
    #for obj in objs:
    #    print(yaml.dump(obj))
