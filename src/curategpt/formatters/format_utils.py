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


def remove_formatting(text: str, expect_format: str = "") -> str:
    """
    Remove markdown formatting from text if present.

    :param text:
    :param expect_format: The expected format of the text, e.g., "json" (optional)
    :return:
    """
    if text.startswith("```" + expect_format):
        return text[3 + len(expect_format) : -3]
    return text
