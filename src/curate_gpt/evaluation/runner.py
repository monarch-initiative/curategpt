import logging
import platform
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TextIO, Union

import yaml
from pydantic import BaseModel

from curate_gpt import BasicExtractor, ChromaDBAdapter
from curate_gpt.agents.dac_agent import DatabaseAugmentedCompletion
from curate_gpt.evaluation.dae_evaluator import DatabaseAugmentedCompletionEvaluator
from curate_gpt.evaluation.evaluation_datamodel import Task
from curate_gpt.evaluation.splitter import stratify_collection, stratify_collection_to_store

logger = logging.getLogger(__name__)


def run_task(
    task: Task, report_path=None, report_file: TextIO = None, fresh=False, **kwargs
) -> Task:
    """
    Evaluate the agent on a test collection.

    :param task:
    :param report_path:
    :param report_file:
    :param fresh: if True, overwrite existing results file
    :param kwargs: passed to the evaluator
    :return: the task with results
    """
    task = deepcopy(task)
    if not task.working_directory:
        raise ValueError("Working directory must be specified")
    wd = Path(task.working_directory)
    wd.mkdir(exist_ok=True, parents=True)
    results_file_path = wd / f"{task.id}.results.yaml"
    if results_file_path.exists():
        logger.info(f"Results file exists at {results_file_path}")
        if not fresh:
            logger.info(f"Loading results from {results_file_path}")
            task = Task.parse_obj(yaml.safe_load(results_file_path.open()))
            return task
        else:
            logger.info("Overwriting existing results file...")
    extractor = BasicExtractor(model_name=task.model_name)
    if task.target_db_path is None:
        task.target_db_path = str(wd / "db")
    target_path = task.target_db_path
    logger.info(f"Stratifying collection {task.source_collection} from {task.source_db_path}")
    db = ChromaDBAdapter(task.source_db_path)
    sc = stratify_collection_to_store(
        db,
        task.source_collection,
        output_path=target_path,
        num_training=task.num_training,
        num_testing=task.num_testing,
        num_validation=task.num_validation,
        embedding_model=task.embedding_model_name,
        force=fresh,
    )
    logger.debug(f"Stratified collection: {sc}")
    tdb = ChromaDBAdapter(target_path)
    # set start time to current time (ISO format)
    task.task_started = str(datetime.now())
    # get current operating system
    task.executed_on = (
        f"{platform.system()}-{platform.release()}-{platform.version()}-{platform.machine()}"
    )
    agent = DatabaseAugmentedCompletion(
        knowledge_source=tdb, knowledge_source_collection="", extractor=extractor
    )
    evaluator = DatabaseAugmentedCompletionEvaluator(
        agent=agent, fields_to_predict=task.fields_to_predict, fields_to_mask=task.fields_to_mask
    )
    if report_path is not None:
        task.report_path = report_path
    if task.report_path is not None:
        report_file = open(task.report_path, "w")
    if report_file is None:
        report_file = open(wd / f"{task.id}.log.yaml", "w")
    report_file.write("## Task\n")
    report_file.write(yaml.dump(task.dict(), sort_keys=False))
    results = evaluator.evaluate(
        test_collection=sc["testing"],
        num_tests=task.num_testing,
        collection=sc["training"],
        report_file=report_file,
        **kwargs,
    )
    task.results = results
    # set finish time to current time (ISO format)
    task.task_finished = str(datetime.now())
    with open(results_file_path, "w") as file:
        file.write(yaml.dump(task.dict(), sort_keys=False))
    return task
