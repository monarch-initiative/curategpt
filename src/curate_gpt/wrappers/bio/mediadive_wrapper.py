import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, Iterator, Optional

import requests_cache
from oaklib import BasicOntologyInterface

from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)

BASE_URL = "https://mediadive.dsmz.de/rest"


@dataclass
class MediaDiveWrapper(BaseWrapper):

    """
    A wrapper over MediaDive.
    """

    name: ClassVar[str] = "mediadive"

    _label_adapter: BasicOntologyInterface = None

    default_object_type = "ComplexMixture"

    def object_ids(self, **kwargs) -> Iterator[str]:
        """
        Get all medium ids in database

        :param kwargs:
        :return:
        """
        session = requests_cache.CachedSession("mediadive")

        response = session.get(f"{BASE_URL}/media")
        response.raise_for_status()
        data = response.json()["data"]
        yield from [obj["id"] for obj in data]

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        """
        All media

        :param collection:
        :param object_ids: optional list of object ids to retrieve (if None, all objects are retrieved)
        :param kwargs:
        :return:
        """
        session = requests_cache.CachedSession("mediadive")

        if not object_ids:
            object_ids = self.object_ids(**kwargs)
        else:
            object_ids = object_ids

        for object_id in object_ids:
            response = session.get(f"{BASE_URL}/medium/{object_id}")
            response.raise_for_status()
            data = response.json()["data"]
            obj = data["medium"]
            # TODO: Finalize after https://github.com/biopragmatics/bioregistry/issues/941
            obj["id"] = f"mediadive.medium:{obj['id']}"
            solutions = data.get("solutions", {})
            if solutions:
                obj["solutions"] = solutions
            else:
                logger.warning(f"No solutions for {object_id}")
            yield obj
