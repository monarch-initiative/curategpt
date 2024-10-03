from curategpt.wrappers import get_wrapper

from tests import INPUT_DIR


def test_linkml_schema_objects():
    wrapper = get_wrapper("linkml_schema")
    wrapper.source_locator = INPUT_DIR / "biolink-model.yaml"
    objs = list(wrapper.objects())
    genes = [o for o in objs if o["name"] == "gene"]
    assert len(genes) == 1
    # for obj in objs:
    #    print(yaml.dump(obj))
