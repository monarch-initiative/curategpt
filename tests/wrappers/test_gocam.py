import json

import pytest
import yaml

from curate_gpt.wrappers.bio.gocam_wrapper import GOCAMWrapper
from tests import INPUT_DIR


@pytest.fixture
def gocam_wrapper() -> GOCAMWrapper:
    view = GOCAMWrapper()
    return view


def test_gocam_object(gocam_wrapper):
    doc = json.load(open(str(INPUT_DIR / "gocam-613aae0000000813.json")))
    obj = gocam_wrapper.object_from_dict(doc)
    print(yaml.dump(obj, sort_keys=False))
    [activity] = [a for a in obj["activities"] if a["gene"] == "PRKAA1"]
    assert activity["activity"] == "ProteinSerineThreonineKinaseActivity"
    [rel] = activity["relationships"]
    assert rel["type"] == "DirectlyNegativelyRegulates"
    assert rel["target_gene"] == "CASP6"
    assert rel["target_activity"] == "CysteineTypeEndopeptidaseActivityInvolvedInApoptoticProcess"


def test_all_objects(gocam_wrapper):
    for obj in gocam_wrapper.objects():
        _ = yaml.dump(obj, sort_keys=False)
