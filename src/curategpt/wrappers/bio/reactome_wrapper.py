"""Chat with a KB."""

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Iterable, Iterator, List, Optional

import requests_cache
from oaklib import BasicOntologyInterface

from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)

BASE_URL = "https://reactome.org/ContentService/data"


def ids_from_tree(objs: List):
    """
    Recursively yield all ids from a tree of objects

    Note: may contain duplicates

    :param objs:
    :return:
    """
    for obj in objs:
        yield "Reactome:" + obj["stId"]
        if "children" in obj:
            yield from ids_from_tree(obj["children"])


def term_object(obj: dict):
    return {
        "id": obj["databaseName"] + ":" + obj["accession"],
        "label": obj["name"],
    }


def pub_object(obj: dict):
    if "pubMedIdentifier" not in obj:
        logger.warning(f"Missing pubmed id for {obj}")
        return None
    return {
        "id": "PMID:" + str(obj["pubMedIdentifier"]),
        "title": obj["title"],
    }


def simple_entity_object(obj: dict):
    return {
        "id": "Reactome:" + str(obj["stId"]),
        "label": obj["displayName"],
        "type": obj.get("referenceType", None),
    }


def generic_object(obj: dict):
    return obj["displayName"]


OBJECT_FUNCTION_MAP = {
    "compartment": term_object,
    "literatureReference": pub_object,
    "catalystActivity": generic_object,
    "input": simple_entity_object,
    "output": simple_entity_object,
    "hasComponent": simple_entity_object,
    "precedingEvent": simple_entity_object,
    "goBiologicalProcess": term_object,
}


@dataclass
class ReactomeWrapper(BaseWrapper):
    """
    A wrapper over a Reactome API.
    """

    name: ClassVar[str] = "reactome"

    prefix = "Reactome"

    _label_adapter: BasicOntologyInterface = None

    default_object_type = "Event"

    taxon_id: str = field(default="NCBITaxon:9606")

    def object_ids(self, taxon_id: str = None, **kwargs) -> Iterator[str]:
        """
        Get all object ids for a given taxon id

        :param taxon_id:
        :param kwargs:
        :return:
        """
        session = requests_cache.CachedSession("reactome")

        if not taxon_id:
            taxon_id = self.taxon_id
        taxon_id_acc = taxon_id.split(":")[1]
        response = session.get(f"{BASE_URL}/eventsHierarchy/{taxon_id_acc}/")
        response.raise_for_status()
        obj = response.json()
        yield from ids_from_tree(obj)

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        """
        All events

        :param collection:
        :param object_ids:
        :param kwargs:
        :return:
        """
        session = requests_cache.CachedSession("reactome")

        if not object_ids:
            object_ids = self.object_ids(**kwargs)
        else:
            object_ids = object_ids

        visited = set()
        for object_id in object_ids:
            if object_id in visited:
                continue
            visited.add(object_id)
            logger.info(f"Getting {object_id}")
            local_id = object_id.split(":")[1]
            response = session.get(f"{BASE_URL}/query/{local_id}")
            obj = response.json()
            summations = obj["summation"]
            new_obj = {
                "id": object_id,
                "label": obj["displayName"],
                "speciesName": obj["speciesName"],
                "description": "\n".join([x["text"] for x in summations]),
                "type": obj["schemaClass"],
            }
            for key, func in OBJECT_FUNCTION_MAP.items():
                if key in obj:
                    new_obj[key] = [func(x) for x in obj[key] if isinstance(x, dict)]
            yield new_obj
