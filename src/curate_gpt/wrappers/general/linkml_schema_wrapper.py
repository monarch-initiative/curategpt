"""Wrapper for LinkML Schema documents."""
import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, Iterator, Optional

from linkml_runtime import SchemaView
from linkml_runtime.dumpers import json_dumper

from curate_gpt.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)


@dataclass
class LinkMLSchemarapper(BaseWrapper):
    """
    A wrapper over linkml schema documents.
    """

    name: ClassVar[str] = "linkml_schema"

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        path = self.source_locator

        sv = SchemaView(str(path))
        yield from self.from_schemaview(sv)

    def from_schemaview(self, sv: SchemaView) -> Iterator[Dict]:
        schema = sv.materialize_derived_schema()
        elt_types = ["classes", "slots", "types", "enums"]
        for elt_type in elt_types:
            for elt in getattr(schema, elt_type, {}).values():
                yield json_dumper.to_dict(elt)
