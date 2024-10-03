from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TextIO

from curategpt.agents.base_agent import BaseAgent
from curategpt.evaluation.evaluation_datamodel import ClassificationMetrics


@dataclass
class BaseEvaluator(ABC):
    """Base class for evaluators."""

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

    @abstractmethod
    def evaluate_object(self, obj, **kwargs) -> ClassificationMetrics:
        """
        Evaluate the agent on a single object.

        :param obj:
        :param kwargs:
        :return:
        """
        raise NotImplementedError
