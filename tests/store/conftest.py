import os
from typing import List

import pytest


@pytest.fixture
def example_texts() -> List[str]:
    return [
        "The quick brown fox jumps over the lazy dog",
        "canine",
        "vulpine",
        "let sleeping dogs lie",
        "chicken",
        "wings",
        "airplane",
    ]


@pytest.fixture
def example_combo_texts() -> List[str]:
    return [
        "pineapple helicopter 1",
        "pineapple helicopter 2",
        "apple helicopter",
        "guava airplane",
        "mango airplane",
        "papaya train",
        "banana train",
        "zucchini firetruck",
        "apple train",
        "orange lorry",
        "orange lorry chimney",
        "orange lorry window",
        "pineapple ship chimney",
        "cheese helicopter",
        "parmesan firefighter",
        "parmesan firefighter 5",
        "cheddar doctor",
        "swiss doctor",
        "helicopter apple",
        "chopper apple",
        "helicopter golden delicious",
    ]


requires_openai_api_key = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None,
    reason="Skipping test: OPENAI_API_KEY environment variable is not set. \
            This test requires an OPENAI_API_KEY to run and is not included in the results.",
)
