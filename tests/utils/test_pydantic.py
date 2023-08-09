from typing import Union, Dict, Any, List


class TestClass:
    object: Union[Dict[str, Any], List[Dict[str, Any]]] = {}


def test_pydantic():
    tc = TestClass()
    tc.object = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    print(tc.object)
    assert tc.object[0]["a"] == 1