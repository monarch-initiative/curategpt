"""Chat with a KB."""

import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, Iterator, Optional

import requests
from curate_gpt.wrappers import BaseWrapper
from oaklib import BasicOntologyInterface, get_adapter

URL = "https://api.microbiomedata.org/biosamples?per_page={limit}&page={cursor}"


def _get_samples_chunk(cursor=1, limit=200) -> dict:
    """
    Get a chunk of samples from NMDC.

    :param cursor:
    :return:
    """
    url = URL.format(limit=limit, cursor=cursor)
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Could not download samples from {url}")


def get_samples(cursor=1, limit=200, maximum: int = None) -> Iterator[dict]:
    """
    Iterate through all samples in NMDC and download them.

    :param cursor:
    :return:
    """
    # note: do NOT do this recursively, as it will cause a stack overflow
    initial_chunk = _get_samples_chunk(cursor=cursor, limit=limit)
    num = initial_chunk["meta"]["count"]
    if maximum is not None:
        num = min(num, maximum)
    yield from initial_chunk["results"]
    while True:
        cursor += 1
        if cursor * limit >= num:
            break
        next_chunk = _get_samples_chunk(cursor=cursor, limit=limit)
        yield from next_chunk["results"]


logger = logging.getLogger(__name__)


@dataclass
class NMDCWrapper(BaseWrapper):
    """
    A wrapper over the NMDC Biosample API.
    """

    name: ClassVar[str] = "nmdc"

    default_object_type = "BioSample"

    _label_adapter: BasicOntologyInterface = None

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        for sample in get_samples():
            for k in ["env_broad_scale", "env_local_scale", "env_medium"]:
                if k in sample:
                    self._labelify(sample[k]["term"])
            yield sample

    @property
    def label_adapter(self) -> BasicOntologyInterface:
        """Get the label adapter."""
        if self._label_adapter is None:
            self._label_adapter = get_adapter("sqlite:obo:envo")
        return self._label_adapter

    def _labelify(self, term: Dict):
        if "label" not in term or not term["label"]:
            term["label"] = self.label_adapter.label(term["id"])
