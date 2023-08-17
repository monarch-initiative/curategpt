"""Utilities for streamlit app."""
from typing import Dict, List, Optional

import yaml

from curate_gpt.conf import EXAMPLES_DIR

HELP_CASES = EXAMPLES_DIR / "bio.yaml"


def get_case_collection():
    with open(HELP_CASES) as stream:
        return yaml.safe_load(stream)


def get_applicable_examples(collection: Optional[str], mode: str, relax=True) -> List[Dict]:
    """
    Get applicable examples for a given collection and mode.

    :param collection:
    :param mode:
    :return:
    """
    if mode:
        mode = mode.upper()
    examples = []
    cases = get_case_collection()["cases"]
    for case in cases:
        if mode and case["mode"] != mode:
            continue
        if collection and case["source"] not in collection:
            # TODO: less hacky check
            continue
        case = {k: v for k, v in case.items() if k not in ["domains", "answers"]}
        examples.append(case)
    if not examples and relax and collection:
        # If no examples are found, try to relax the collection
        return get_applicable_examples(None, mode, relax=False)
    return examples
