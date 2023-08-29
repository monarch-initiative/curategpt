import logging
import shutil

import pytest
from oaklib import get_adapter

from curate_gpt import ChromaDBAdapter
from curate_gpt.extract import BasicExtractor
from curate_gpt.wrappers import get_wrapper
from curate_gpt.wrappers.ontology.ontology_wrapper import OntologyWrapper
from tests import INPUT_DIR, OUTPUT_DIR



def test_filesystem_objects():
    wrapper = get_wrapper("filesystem")
    for obj in wrapper.objects():
        print(obj)
