import logging

import pytest
import yaml
from curate_gpt.wrappers.clinical.hpoa_wrapper import HPOAWrapper

from tests import INPUT_DIR, OUTPUT_DIR

TEMP_DB = OUTPUT_DIR / "obj_tmp"


logger = logging.getLogger(__name__)


@pytest.mark.parametrize("group_by_publication", [True, False])
def test_hpoa(group_by_publication):
    wrapper = HPOAWrapper(group_by_publication=group_by_publication)
    with open(INPUT_DIR / "example-phenotype-hpoa.tsv") as file:
        vars = list(wrapper.objects_from_file(file))
        print(yaml.dump(vars))
