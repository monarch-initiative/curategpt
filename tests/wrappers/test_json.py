from curate_gpt.wrappers import get_wrapper
from tests import INPUT_DIR


def test_json_objects():
    wrapper = get_wrapper("json")
    wrapper.source_locator = INPUT_DIR / "biolink-model.yaml"
    wrapper.path_expression = "$.classes[*]"
    objs = list(wrapper.objects())
    assert len(objs) == 1
    assert "gene" in objs[0]
