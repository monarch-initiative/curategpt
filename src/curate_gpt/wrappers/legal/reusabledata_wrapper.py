"""Chat with a KB."""
import gzip
import logging
import os
from dataclasses import dataclass, field
from glob import glob
from typing import ClassVar, Dict, Iterable, Iterator, Optional

import requests
import requests_cache
import yaml
from bs4 import BeautifulSoup
from oaklib import BasicOntologyInterface, get_adapter

from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)


@dataclass
class ReusableDataWrapper(BaseWrapper):
    """
    A wrapper over a reusabledata.org yaml files.

    Each yaml file is treated as a distinct dataset object;
    the structure is not altered, other than adding a ``license_text``
    field that is obtained by querying the license_link URL.

    This text can be used for evaluating automated metadata extraction.

    This is a static wrapper: it does not allow search,
    it is expected that all objects from the source are ingested.
    """

    name: ClassVar[str] = "reusabledata"

    _label_adapter: BasicOntologyInterface = None

    default_object_type = "Dataset"

    glob: bool = False

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        session = requests_cache.CachedSession("reusabledata")

        path = self.source_locator or "https://reusabledata.org/data.json"
        objs = requests.get(path).json()
        for obj in objs:
            obj_id = obj["id"]
            license_link = obj.get("license-link", None)
            if license_link:
                if license_link in [
                    "TODO",
                    "https://",
                    "inconsistent",
                    "https://civic.genome.wustl.edu/about",
                    "http://www.supfam.org/about",
                    "ftp://ftp.nextprot.org/pub/README",
                    "ftp://ftp.jcvi.org/pub/data/TIGRFAMs/COPYRIGHT",
                ]:
                    logger.warning(f"base link {license_link} for {obj_id}")
                    continue
                if license_link.startswith("ftp://"):
                    license_link = license_link.replace("ftp://", "http://")
                logger.info(f"{obj_id} license url: {license_link}")
                response = session.get(license_link)
                if not response.ok:
                    logger.warning(f"bad link {license_link} for {obj_id}")
                    continue
                data = response.text
                soup = BeautifulSoup(data, "html.parser")
                license_text = soup.get_text()
                obj["license_text"] = license_text
            yield obj
