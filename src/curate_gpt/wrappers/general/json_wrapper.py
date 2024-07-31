"""Wrapper for JSON (or YAML) documents."""

import json
import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, Iterator, Optional

import yaml
from jsonpath_ng import parse

from curate_gpt.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)


@dataclass
class JSONWrapper(BaseWrapper):
    """
    A wrapper over a json (or yaml) document.

    Uses json path expressions

    This is a static wrapper: it cannot be searched
    """

    name: ClassVar[str] = "json"
    format: str = None
    path_expression: str = None

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        path = self.source_locator

        with open(path) as file:
            print(f"Loading {path}")
            format = self.format or str(path).split(".")[-1]
            if format == "json":
                obj = json.load(file)
            elif format == "yaml":
                obj = yaml.safe_load(file)
            else:
                raise ValueError(f"Unknown format {format}")
        yield from self.wrap_object(obj)

    def wrap_object(self, obj: Dict) -> Iterator[Dict]:
        jsonpath_expression = parse(self.path_expression)
        for match in jsonpath_expression.find(obj):
            yield match.value
