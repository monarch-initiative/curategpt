"""Chat with a KB."""
import logging
from dataclasses import dataclass, field
from time import sleep
from typing import ClassVar, Dict, Iterable, Iterator, Optional

import requests_cache

from curate_gpt.wrappers import BaseWrapper

URL = "https://fusion.ess-dive.lbl.gov/deepdive?rowStart={cursor}&pageSize={limit}"


def _get_records_chunk(session: requests_cache.CachedSession, cursor=1, limit=200) -> dict:
    """
    Get a chunk of records from ESSDeepDive.

    :param cursor:
    :return:
    """
    url = URL.format(limit=limit, cursor=cursor)
    response = session.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Could not download records from {url}")


def get_records(
    session: requests_cache.CachedSession, cursor=1, limit=200, maximum: int = None
) -> Iterator[dict]:
    """
    Iterate through all records in ESSDeepDive and download them.

    :param cursor:
    :return:
    """
    # note: do NOT do this recursively, as it will cause a stack overflow
    initial_chunk = _get_records_chunk(session, cursor=cursor, limit=limit)
    yield from initial_chunk["results"]
    while True:
        cursor += 1
        next_chunk = _get_records_chunk(session, cursor=cursor, limit=limit)
        yield from next_chunk["results"]
        pc = next_chunk["pageCount"]
        logger.warning(f"Got {pc} pages")
        if not pc:
            break
        sleep(0.1)


logger = logging.getLogger(__name__)


@dataclass
class ESSDeepDiveWrapper(BaseWrapper):

    """
    A wrapper over the ESSDeepDive API.
    """

    name: ClassVar[str] = "ess_deepdive"

    default_object_type = "Class"

    session: requests_cache.CachedSession = field(
        default_factory=lambda: requests_cache.CachedSession("ess_deepdive")
    )

    limit: int = field(default=50)

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        seen = set()
        for record in get_records(self.session, limit=self.limit):
            id = record["data_file_url"] + "-" + record["field_name"]
            if id in seen:
                logger.warning(f"Skipping duplicate record {id}")
                continue
            seen.add(id)
            record["id"] = id
            yield record
