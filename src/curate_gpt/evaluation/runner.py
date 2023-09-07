import logging
import platform
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TextIO

import yaml

from curate_gpt import BasicExtractor, ChromaDBAdapter
from curate_gpt.agents.dac_agent import DatabaseAugmentedCompletion
from curate_gpt.evaluation.dae_evaluator import DatabaseAugmentedCompletionEvaluator
from curate_gpt.evaluation.evaluation_datamodel import Task
from curate_gpt.evaluation.splitter import stratify_collection_to_store

logger = logging.getLogger(__name__)


def run_task(
    task: Task,
    report_path=None,
    report_file: TextIO = None,
    report_tsv_file: TextIO = None,
    fresh=False,
    **kwargs,
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
    if not target_path:
        raise ValueError("Target database path must be specified")
    logger.info(f"Stratifying collection {task.source_collection} from {task.source_db_path}")
    db = ChromaDBAdapter(task.source_db_path)
    if task.stratified_collection:
        sc = task.stratified_collection
        sc_dict = {
            "training": sc.training_set_collection,
            "testing": sc.testing_set_collection,
        }
        logger.debug(f"Stratified collection: {sc}")
    else:
        sc_dict = stratify_collection_to_store(
            db,
            task.source_collection,
            output_path=target_path,
            num_training=task.num_training,
            num_testing=task.num_testing,
            num_validation=task.num_validation,
            embedding_model=task.embedding_model_name,
            force=fresh,
        )
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
    if task.additional_collections:
        if len(task.additional_collections) > 1:
            raise NotImplementedError("Only one additional collection is supported")
        agent.document_adapter = tdb
        agent.document_adapter_collection = task.additional_collections[0]
    # TODO: use the task object directly in the evaluator
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
    if report_tsv_file is None:
        report_tsv_file = open(wd / f"{task.id}.results.tsv", "w")
        commented_yaml = yaml.dump(task.dict(), sort_keys=False)
        lines = [f"# {line}\n" for line in commented_yaml.splitlines()]
        report_tsv_file.write("".join(lines))
    logger.info(f"Evaluating agent {agent} on collection {sc_dict['testing']}")
    results = evaluator.evaluate(
        test_collection=sc_dict["testing"],
        num_tests=task.num_testing,
        collection=sc_dict["training"],
        report_file=report_file,
        report_tsv_file=report_tsv_file,
        generate_background=task.generate_background,
        **kwargs,
    )
    logger.info("Collecting results...")
    task.results = results
    # set finish time to current time (ISO format)
    task.task_finished = str(datetime.now())
    with open(results_file_path, "w") as file:
        file.write(yaml.dump(task.dict(), sort_keys=False))
    return task
