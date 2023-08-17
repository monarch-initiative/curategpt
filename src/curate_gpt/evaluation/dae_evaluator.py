import logging
from dataclasses import dataclass
from typing import List, TextIO

import yaml

from curate_gpt.agents.dae_agent import DatabaseAugmentedExtractor
from curate_gpt.evaluation.base_evaluator import BaseEvaluator
from curate_gpt.utils.metrics import (
    ClassificationMetrics,
    ClassificationOutcome,
    aggregate_metrics,
    calculate_metrics,
    evaluate_predictions,
)

logger = logging.getLogger(__name__)


@dataclass
class DatabaseAugmentedExtractorEvaluator(BaseEvaluator):
    """
    Retrieves objects in response to a query using a structured knowledge source.
    """

    agent: DatabaseAugmentedExtractor = None
    hold_back_fields: List[str] = None

    def evaluate(
        self, test_collection: str, num_tests=10000, report_file: TextIO = None, **kwargs
    ) -> ClassificationMetrics:
        agent = self.agent
        db = agent.knowledge_source
        test_objs = db.peek(collection=test_collection, limit=num_tests)
        all_metrics = []
        if not test_objs:
            raise ValueError(f"No test objects found in collection {test_collection}")
        for test_obj in test_objs:
            query_obj = {k: v for k, v in test_obj.items() if k not in self.hold_back_fields}
            logger.debug(f"## Query: {query_obj}")
            ao = agent.generate_extract(query_obj, **kwargs)
            logger.debug(f"## Expected: {test_obj}")
            logger.debug(f"## Prediction: {ao.object}")
            outcomes = []
            for f in self.hold_back_fields:
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
            if report_file:
                report_file.write("# RESULTS\n")
                report_file.write(f"## Query:\n{yaml.dump(query_obj)}\n")
                report_file.write(f"## Prompt:\n{ao.annotations.get('prompt', None)}\n")
                report_file.write(f"## Expected:\n{yaml.dump(test_obj)}\n")
                report_file.write(f"## Predicted:\n{yaml.dump(ao.object)}\n")
                report_file.write(f"## Metrics:\n{yaml.dump(metrics.dict())}\n")
                report_file.write(f"## Aggregated:\n{yaml.dump(aggregated.dict())}\n")
        aggregated = aggregate_metrics(all_metrics)
        return aggregated
