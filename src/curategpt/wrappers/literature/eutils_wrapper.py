"""Chat with a KB."""

import logging
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

import requests
import requests_cache
import xmltodict
from eutils import Client

from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{tool}.fcgi"


@dataclass
class EUtilsWrapper(BaseWrapper, ABC):
    """
    A wrapper to provide a search facade over NCBI Biosample.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
    """

    name: ClassVar[str] = "__eutils__"

    eutils_db: ClassVar[str] = None

    id_prefix: ClassVar[str] = None

    fetch_tool: ClassVar[str] = "efetch"

    eutils_client: Client = None

    session: requests.Session = field(default_factory=lambda: requests.Session())

    _uses_cache: bool = False

    def set_cache(self, name: str) -> None:
        self.session = requests_cache.CachedSession(name)
        self._uses_cache = True

    def external_search(self, text: str, expand: bool = True, **kwargs) -> List[Dict]:
        db = self.eutils_db
        if expand:
            logger.info(f"Expanding search term: {text} to create {db} query")
            model = self.extractor.model
            response = model.prompt(
                text, system="generate a semi-colon separated list of the most relevant terms"
            )
            terms = response.text().split(";")
            search_term = " OR ".join(terms)
        else:
            search_term = text
        logger.info(f"Constructed search term: {search_term}")
        # Parameters for the request
        params = {
            "db": db,
            "term": search_term,
            "retmax": 100,
            "sort": "relevance",
            "retmode": "json",
        }

        time.sleep(0.5)
        response = requests.get(ESEARCH_URL, params=params)
        data = response.json()

        # Extract IDs from the response
        ids = data["esearchresult"]["idlist"]

        if not ids:
            logger.warning(f"No results with {search_term}")
            if expand:
                logger.info("Trying again without expansion")
                return self.external_search(text, expand=False, **kwargs)
            else:
                logger.error(f"Failed to find results for {text}")
                return []

        logger.info(f"Found {len(ids)} results: {ids}")
        return self.objects_by_ids(ids)

    def objects_by_ids(self, object_ids: List[str]) -> List[Dict]:
        id_prefix = self.id_prefix or self.eutils_db
        object_ids = sorted([x.replace(f"{id_prefix}:", "") for x in object_ids])
        session = self.session
        logger.debug(f"Using session: {session} [cached: {self._uses_cache} for {object_ids}")

        # Parameters for the efetch request
        efetch_params = {
            "db": self.eutils_db,
            "id": ",".join(object_ids),  # Combine  IDs into a comma-separated string
            "retmode": "xml",
        }
        # if not self._uses_cache:
        time.sleep(0.25)
        tool = self.fetch_tool
        efetch_response = session.get(EFETCH_URL.format(tool=tool), params=efetch_params)
        if not efetch_response.ok:
            logger.error(f"Failed to fetch data for {object_ids}")
            raise ValueError(
                f"Failed to fetch data for {object_ids} using {session} and {efetch_params}"
            )
        results = xmltodict.parse(efetch_response.text)
        return self.objects_from_dict(results)

    def objects_from_dict(self, results: Dict) -> List[Dict]:
        raise NotImplementedError
