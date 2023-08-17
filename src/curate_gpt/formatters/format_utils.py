from typing import Dict

import inflection
import yaml


def camelify(text: str) -> str:
    """
    Convert text to camel case.

    :param text:
    :return:
    """
    # replace all non-alphanumeric characters with underscores
    safe = "".join([c if c.isalnum() else "_" for c in text])
    return inflection.camelize(safe)


def object_as_yaml(obj: Dict) -> str:
    """
    Canonical YAML representation of an object.

    - no empty values
    - no sorting (default insertion order)

    :param obj:
    :return:
    """
    return yaml.dump({k: v for k, v in obj.items() if v}, sort_keys=False)
