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
class BacDiveWrapper(BaseWrapper):
    """
    A wrapper over a bacdive json files.

    This is a static wrapper: it cannot be searched
    """

    name: ClassVar[str] = "bacdive"
    prefix = "bacdive"
    default_object_type = "OrganismTaxon"
    format: str = None
    path_expression: str = None

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        path = self.source_locator

        with open(path) as file:
            data = json.load(file)
            for obj in data.values():
                yield from self.from_object(obj)

    def from_object(self, obj: Dict) -> Iterator[Dict]:
        general = obj["General"]
        name_info = obj["Name and taxonomic classification"]
        new_obj = {}
        new_obj["id"] = self.create_curie(general['BacDive-ID'])
        new_obj["name"] = name_info.get("full scientific name", None)
        if not new_obj["name"]:
            new_obj["name"] = name_info["LPSN"].get("scientific name", None)
        taxs = general.get("NCBI tax id", None)
        if not isinstance(taxs, list):
            taxs = [taxs]
        while taxs:
            tax = taxs.pop()
            if tax:
                tax_id = tax.get("NCBI tax id", None)
                new_obj["taxon"] = f"NCBITaxon:{tax_id}"
                break
        new_obj = {**new_obj, **obj}
        yield new_obj

