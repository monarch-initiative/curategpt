"""Base Agent."""

from abc import ABC
from dataclasses import dataclass
from typing import Union

from curate_gpt import DBAdapter, Extractor
from curate_gpt.wrappers import BaseWrapper


@dataclass
class BaseAgent(ABC):
    """
    Base class for agents.

    An agent is capable of composing together different actions to achieve a goal.

    An agent typically has a *knowledge source* that is uses to search for information.
    An agent also has access to a model through an extractor.
    """

    knowledge_source: Union[DBAdapter, BaseWrapper] = None
    """A searchable source of information"""

    knowledge_source_collection: str = None

    extractor: Extractor = None
    """Engine performing LLM operations, including extracting from prompt responses"""
