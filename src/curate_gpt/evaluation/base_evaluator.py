from abc import ABC
from dataclasses import dataclass
from typing import TextIO

from curate_gpt.agents.base_agent import BaseAgent
from curate_gpt.evaluation.evaluation_datamodel import ClassificationMetrics


@dataclass
class BaseEvaluator(ABC):
    """
    Base class for evaluators.
    """

    agent: BaseAgent = None

    def evaluate(
        self, test_collection: str, num_tests=10000, report_file: TextIO = None, **kwargs
    ) -> ClassificationMetrics:
        """
        Evaluate the agent on a test collection.

        :param test_collection:
        :param num_tests:
        :param report_file:
        :param kwargs:
        :return:
        """
        raise NotImplementedError
