import logging
from dataclasses import dataclass, field
from typing import List, TextIO

import yaml

from curate_gpt.agents.dac_agent import DatabaseAugmentedCompletion
from curate_gpt.evaluation.base_evaluator import BaseEvaluator
from curate_gpt.evaluation.calc_statistics import (
    aggregate_metrics,
    calculate_metrics,
    evaluate_predictions,
)
from curate_gpt.evaluation.evaluation_datamodel import ClassificationMetrics, ClassificationOutcome

logger = logging.getLogger(__name__)


@dataclass
class DatabaseAugmentedCompletionEvaluator(BaseEvaluator):
    """
    Retrieves objects in response to a query using a structured knowledge source.
    """

    agent: DatabaseAugmentedCompletion = None
    fields_to_predict: List[str] = field(default_factory=list)
    fields_to_mask: List[str] = field(default_factory=list)

    def evaluate(
        self,
        test_collection: str,
        num_tests: int = None,
        report_file: TextIO = None,
        working_directory: str = None,
        **kwargs,
    ) -> ClassificationMetrics:
        """
        Evaluate the agent on a test collection.

        Note: the main collection used for few-shot learning is passed in kwargs (this may change)

        :param test_collection:
        :param num_tests:
        :param report_file:
        :param kwargs:
        :return:
        """
        agent = self.agent
        db = agent.knowledge_source
        # TODO: use get()
        test_objs = list(db.peek(collection=test_collection, limit=num_tests))
        if any(obj for obj in test_objs if any(f not in obj for f in self.fields_to_predict)):
            logger.info("Alternate strategy to get test objs; query whole collection")
            test_objs = db.peek(collection=test_collection, limit=1000000)
            test_objs = [obj for obj in test_objs if all(f in obj for f in self.fields_to_predict)]
            test_objs = test_objs[:num_tests]
        if num_tests and len(test_objs) < num_tests:
            raise ValueError(
                f"Insufficient test objects in collection {test_collection}; "
                f"{len(test_objs)} < {num_tests}"
            )
        all_metrics = []
        if not test_objs:
            raise ValueError(f"No test objects found in collection {test_collection}")
        n = 0
        for test_obj in test_objs:
            test_obj_query = {
                k: v
                for k, v in test_obj.items()
                if k not in self.fields_to_predict and k not in self.fields_to_mask
            }
            logger.debug(f"## Query: {test_obj_query}")
            ao = agent.complete(test_obj_query, **kwargs)
            logger.debug(f"## Expected: {test_obj}")
            logger.debug(f"## Prediction: {ao.object}")
            outcomes = []
            for f in self.fields_to_predict:
                outcomes.extend(
                    list(evaluate_predictions(ao.object.get(f, None), test_obj.get(f, None)))
                )
            for outcome, info in outcomes:
                if outcome != ClassificationOutcome.TRUE_POSITIVE:
                    logger.debug(f"## Diff: {outcome} - {info}")
            metrics = calculate_metrics(outcomes)
            all_metrics.append(metrics)
            logger.info(f"## Metrics: {metrics.json()}")
            aggregated = aggregate_metrics(all_metrics)
            logger.info(f"## Aggregated: {aggregated.json()}")
            n += 1
            if report_file:
                report_file.write(f"# RESULTS {n}\n")
                report_file.write(f"## Query:\n{yaml.dump(test_obj_query)}\n---\n")
                report_file.write(f"## Prompt:\n{ao.annotations.get('prompt', None)}\n---\n")
                report_file.write(f"## Expected:\n{yaml.dump(test_obj)}\n---\n")
                report_file.write(f"## Predicted:\n{yaml.dump(ao.object)}\n---\n")
                report_file.write(f"## Metrics:\n{yaml.dump(metrics.dict())}\n---\n")
                report_file.write(f"## Aggregated:\n{yaml.dump(aggregated.dict())}\n---\n")
        aggregated = aggregate_metrics(all_metrics)
        return aggregated
