import logging
import random
from typing import Dict, List, Union, Optional, Any

from curate_gpt import ChromaDBAdapter, DBAdapter
from curate_gpt.evaluation.evaluation_datamodel import StratifiedCollection

logger = logging.getLogger(__name__)


def stratify_collection(
    store: DBAdapter,
    collection: str,
    num_training: Optional[int] = None,
    num_testing: Optional[int] = None,
    num_validation=0,
    testing_identifiers: Optional[List[str]] = None,
    fields_to_predict: Optional[Union[str, List[str]]] = None,
    ratio=0.7,
    where: Optional[Dict[str, Any]] = None,
) -> StratifiedCollection:
    """
    Stratifies a collection into training, testing, and validation sets.

    :param store:
    :param collection:
    :param num_training:
    :param num_testing:
    :param num_validation:
    :param fields_to_predict:
    :param ratio:
    :param where:
    :return:
    """
    if where is None:
        where = {}
    cm = store.collection_metadata(collection, include_derived=True)
    size = cm.object_count
    if testing_identifiers is None:
        if num_training is None and num_testing is None:
            num_training = int(size * ratio)
        if not num_testing:
            num_testing = size - num_training
        if not num_training:
            raise ValueError("Must specify either num_training or num_test")
        if num_training + num_testing > size:
            raise ValueError("num_training + num_test must be less than size")
    # objs = list(store.peek(collection, limit=num_training + num_testing + num_validation))
    logger.info(f"Stratifying collection {collection} where={where}")
    if where:
        objs = list(store.find(where=where, collection=collection, limit=1000000))
    else:
        # chromadb doesn't do well with large limits on open-ended queries
        objs = list(store.peek(collection, limit=1000000))
    if fields_to_predict:
        if isinstance(fields_to_predict, str):
            fields_to_predict = fields_to_predict.split(",")
        objs = [obj for obj in objs if all(f in obj for f in fields_to_predict)]
    # randomize order w shuffle
    random.shuffle(objs)
    if testing_identifiers:
        logger.info(f"Taking from {len(testing_identifiers)} identifiers for testing set")
        def _is_in_test_set(obj: dict) -> bool:
            for fld in [store.identifier_field(collection), "original_id"]:
                if fld in obj and obj[fld] in testing_identifiers:
                    return True
            return False
        testing_set = [obj for obj in objs if _is_in_test_set(obj)]
        if not testing_set:
            raise ValueError(f"No testing objects found with identifiers: {testing_identifiers}")
        training_set = [obj for obj in objs if not _is_in_test_set(obj)]
        if num_testing:
            if len(testing_set) < num_testing:
                raise ValueError(f"Only {len(testing_set)} testing objects found with identifiers: {testing_identifiers}")
            testing_set = testing_set[:num_testing]
            logging.info(f"Using {len(testing_set)} testing objects")
        validation_set = []
    else:
        training_set = objs[:num_training]
        testing_set = objs[num_training : num_training + num_testing]
        validation_set = objs[num_training + num_testing : num_training + num_testing + num_validation]
    logger.info(f"Training set size: {len(training_set)}")
    logger.info(f"Testing set size: {len(testing_set)}")
    logger.info(f"Validation set size: {len(validation_set)}")
    sc = StratifiedCollection(
        source=collection,
        training_set=training_set,
        testing_set=testing_set,
        validation_set=validation_set,
    )
    return sc


def stratify_collection_to_store(
    store: DBAdapter, collection: str, output_path: str, embedding_model=None, force=False, **kwargs
) -> Dict[str, str]:
    """
    Stratifies a collection into training, testing, and validation sets.

    Each collection is persisted to a separate collection in the output_path.

    :param store:
    :param collection:
    :param output_path:
    :param embedding_model:
    :param force:
    :param kwargs:
    :return:
    """
    sc = stratify_collection(store, collection=collection, **kwargs)
    output_db = ChromaDBAdapter(output_path)
    collections = {}
    existing_collections = output_db.list_collection_names()
    for sn in ["training", "testing", "validation"]:
        objs = getattr(sc, f"{sn}_set", [])
        size = len(objs)
        cn = f"{collection}_{sn}_{size}"
        collections[sn] = cn
        logging.info(f"Writing {size} objects to {cn}")
        if cn in existing_collections:
            logger.info(f"Collection {cn} already exists")
            if not force:
                logger.info(f"Reusing existing collection {cn}")
                continue
            else:
                logger.info(f"Overwriting existing collection {cn}")
        output_db.remove_collection(cn, exists_ok=True)
        output_db.insert(objs, collection=cn, model=embedding_model)
    return collections
