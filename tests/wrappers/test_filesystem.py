from pathlib import Path

from curate_gpt.wrappers import get_wrapper

THIS = Path(__file__).name


def test_filesystem_objects():
    root = Path(__file__).absolute().parent
    wrapper = get_wrapper("filesystem", root_directory=root)
    objs = list(wrapper.objects())
    assert len(objs) > 1
    assert len([obj for obj in objs if obj["name"] == THIS]) == 1
