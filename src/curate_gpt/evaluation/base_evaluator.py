from abc import ABC
from dataclasses import dataclass

from curate_gpt.agents.base_agent import BaseAgent


@dataclass
class BaseEvaluator(ABC):
    """
    Base class for evaluators.
    """

    agent: BaseAgent = None
