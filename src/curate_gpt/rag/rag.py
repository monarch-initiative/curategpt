"""Retrieval Augmented Generation (RAG) Base Class."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Type

from pydantic import BaseModel as BaseModel

from curate_gpt.store.db_adapter import DBAdapter

logger = logging.getLogger(__name__)

OBJECT = Dict[str, Any]


@dataclass
class RAG(ABC):
    db_adapter: DBAdapter = None
    root_class: Type[BaseModel] = None

    @abstractmethod
    def generate(self, text: str, **kwargs) -> OBJECT:
        """
        Perform Retrieval Augmented Generation (RAG) on text

        :param text:
        :param kwargs:
        :return:
        """
