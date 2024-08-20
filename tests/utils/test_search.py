import numpy as np
import pytest

from curate_gpt.utils.vector_algorithms import mmr_diversified_search

vectors = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
)


@pytest.mark.parametrize(
    "query_vector,relevance_factor,expected",
    [
        ([1, 1, 1], 1.0, [4, 3, 0, 1, 2]),  # most relevant first
        ([1, 0.5, 0], 0.5, [3, 0, 2, 4, 1]),  # balanced tradeoff
        ([1, 0.5, 0], 0.0, [0, 1, 2, 4, 3]),
        ([1, 0.5, 0], 1.0, [3, 0, 4, 1, 2]),  # most relevant first
    ],
)
def test_mmr(query_vector, relevance_factor, expected):
    query_vector = np.array(query_vector)
    diversified_order = mmr_diversified_search(
        query_vector, vectors, relevance_factor=relevance_factor
    )
    assert diversified_order == expected
