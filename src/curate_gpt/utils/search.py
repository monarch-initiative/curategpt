import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def mmr_diversified_search(
    query_vector: np.ndarray, document_vectors: List[np.ndarray], relevance_factor=0.5, top_n=None
) -> List[int]:
    """
    Perform diversified search using Maximal Marginal Relevance (MMR).

    Parameters:
    - query_vector: The vector representing the query.
    - document_vectors: The vectors representing the documents.
    - lambda_: Balance parameter between relevance and diversity.
    - top_n: Number of results to return. If None, return all.

    Returns:
    - List of indices representing the diversified order of documents.
    """

    # If no specific number of results is specified, return all
    if top_n is None:
        top_n = len(document_vectors)

    if top_n == 0:
        return []

    # Calculate cosine similarities between query and all documents
    norms_query = np.linalg.norm(query_vector)
    norms_docs = np.linalg.norm(document_vectors, axis=1)
    similarities = np.dot(document_vectors, query_vector) / (norms_docs * norms_query)

    # Initialize set of selected indices and results list
    selected_indices = set()
    result_indices = []

    # Diversified search loop
    for _ in range(top_n):
        max_mmr = float("-inf")
        best_index = None

        # Loop over all documents
        for idx, _doc_vector in enumerate(document_vectors):
            if idx not in selected_indices:
                relevance = relevance_factor * similarities[idx]
                diversity = 0

                # Penalize based on similarity to already selected documents
                if selected_indices:
                    max_sim_to_selected = max(
                        [
                            np.dot(document_vectors[idx], document_vectors[s])
                            / (
                                np.linalg.norm(document_vectors[idx])
                                * np.linalg.norm(document_vectors[s])
                            )
                            for s in selected_indices
                        ]
                    )
                    diversity = (1 - relevance_factor) * max_sim_to_selected

                mmr_score = relevance - diversity

                # Update best MMR score and index
                if mmr_score > max_mmr:
                    max_mmr = mmr_score
                    best_index = idx

        # Add the best document to the result and mark it as selected
        if best_index is None:
            logger.warning(f"No best index found over {len(document_vectors)} documents.")
            continue
        result_indices.append(best_index)
        selected_indices.add(best_index)

    return result_indices
