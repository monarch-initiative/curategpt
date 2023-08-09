"""Base class for views."""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator

from pydantic import BaseModel


@dataclass
class View:
    adapter = None

    @abstractmethod
    def objects(self) -> Iterator[BaseModel]:
        """
        Returns all objects in the view
        """
        raise NotImplementedError

    @property
    def text_field(self) -> Callable:
        raise NotImplementedError
