import json

import pytest
import yaml

from curate_gpt.wrappers.sysbio.gocam_wrapper import GOCAMWrapper
from tests import INPUT_DIR


@pytest.fixture
def gocam_wrapper() -> GOCAMWrapper:
    view = GOCAMWrapper()
    return view


def test_gocam_object(gocam_wrapper):
    doc = json.load(open(str(INPUT_DIR / "gocam-613aae0000000813.json")))
    obj = gocam_wrapper.object_from_dict(doc)
    print(yaml.dump(obj, sort_keys=False))


def test_all_objects(gocam_wrapper):
    for obj in gocam_wrapper.objects():
        _ = yaml.dump(obj, sort_keys=False)
