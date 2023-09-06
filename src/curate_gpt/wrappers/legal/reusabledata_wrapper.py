"""Chat with a KB."""
import gzip
import os
import requests
import logging
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Iterable, Iterator, Optional

import requests_cache
from bs4 import BeautifulSoup
from glob import glob

import yaml
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
        path = self.source_locator or "."
        if self.glob:
            files = glob.glob(os.path.join(path, "**", self.glob), recursive=True)
        else:
            files = []
            for dirpath, _dirnames, filenames in os.walk(path):
                for filename in filenames:
                    files.append(os.path.join(dirpath, filename))
        for file_name in files:
            if not file_name.endswith(".yaml"):
                continue
            obj = yaml.safe_load(open(file_name))
            license_link = obj.get("license-link", None)
            if license_link:
                if license_link in ["TODO", "https://", "inconsistent", "https://civic.genome.wustl.edu/about", "https://www.supfam.org/about", "ftp://ftp.nextprot.org/pub/README", "ftp://ftp.jcvi.org/pub/data/TIGRFAMs/COPYRIGHT"]:
                    logger.warning(f"base link {license_link} for {file_name}")
                    continue
                if license_link.startswith("ftp://"):
                    license_link = license_link.replace("ftp://", "http://")
                logger.info(f"{file_name} license url: {license_link}")
                response = session.get(license_link)
                if not response.ok:
                    logger.warning(f"bad link {license_link} for {file_name}")
                    continue
                data = response.text
                soup = BeautifulSoup(data, 'html.parser')
                license_text = soup.get_text()
                obj["license_text"] = license_text
            yield obj

