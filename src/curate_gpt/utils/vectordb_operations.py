import logging
from typing import Iterator, Tuple

from curate_gpt import DBAdapter
from curate_gpt.utils.vector_algorithms import compute_cosine_similarity, top_matches

logger = logging.getLogger(__name__)


def match_collections(
    db: DBAdapter, left_collection: str, right_collection: str, other_db: DBAdapter = None
) -> Iterator[Tuple[dict, dict, float]]:
    """
    Match every element in left collection with every element in right collection.

    Currently this returns best matches for left collection only

    :param db:
    :param left_collection:
    :param right_collection:
    :param other_db: optional - defaults to main
    :return: tuple of object pair plus cosine similarity score
    """
    if not other_db:
        other_db = db
    include = ["metadatas", "documents", "embeddings"]
    logger.info(f"Querying left objects from {left_collection}")
    left_objs = list(db.find({}, collection=left_collection, include=include))
    logger.info(f"Querying right objects from {right_collection}")
    right_objs = list(other_db.find({}, collection=right_collection, include=include))
    left_vectors = [info["_embeddings"] for _, __, info in left_objs]
    right_vectors = [info["_embeddings"] for _, __, info in right_objs]
    logger.info(f"Computing cosine similarity for {len(left_vectors)} x {len(right_vectors)}")
    sim_matrix = compute_cosine_similarity(left_vectors, right_vectors)
    logger.info(f"Finding top matches")
    tm_ix, tm_vals = top_matches(sim_matrix)
    logger.info(f"Yielding {len(tm_ix)} matches")
    i = 0
    for ix, val in zip(tm_ix, tm_vals):
        yield left_objs[i][0], right_objs[ix][0], val
        i += 1
