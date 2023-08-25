"""Chat with a KB."""
import logging
import time
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

import requests
import requests_cache
from eutils import Client

from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

RATE_LIMIT_DELAY = 1.0


# TODO: rewrite to subclass EUtilsWrapper
@dataclass
class PubmedWrapper(BaseWrapper):
    """
    A wrapper to provide a search facade over PubMed.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
    """

    name: ClassVar[str] = "pubmed"

    eutils_client: Client = None

    session: requests.Session = field(default_factory=lambda: requests.Session())

    _uses_cache: bool = False

    def set_cache(self, name: str) -> None:
        self.session = requests_cache.CachedSession(name)
        self._uses_cache = True

    def external_search(self, text: str, expand: bool = True, **kwargs) -> List:
        if expand:
            logger.info(f"Expanding search term: {text} to create pubmed query")
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
            "db": "pubmed",
            "term": search_term,
            "retmax": 100,
            "sort": "relevance",
            "retmode": "json",
        }

        # Note: we don't cache this call as there could be many
        # different search terms
        response = requests.get(ESEARCH_URL, params=params)
        time.sleep(RATE_LIMIT_DELAY)
        data = response.json()

        # Extract PubMed IDs from the response
        pubmed_ids = data["esearchresult"]["idlist"]

        if not pubmed_ids:
            logger.warning(f"No results with {search_term}")
            if expand:
                logger.info("Trying again without expansion")
                return self.external_search(text, expand=False, **kwargs)
            else:
                logger.error(f"Failed to find results for {text}")
                return []

        logger.info(f"Found {len(pubmed_ids)} results: {pubmed_ids}")
        return self.objects_by_ids(pubmed_ids)

    def objects_by_ids(self, object_ids: List[str]) -> List[Dict]:
        pubmed_ids = sorted([x.replace("PMID:", "") for x in object_ids])
        session = self.session
        logger.debug(f"Using session: {session} [cached: {self._uses_cache} for {pubmed_ids}")

        # Parameters for the efetch request
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(pubmed_ids),  # Combine PubMed IDs into a comma-separated string
            "rettype": "medline",
            "retmode": "text",
        }
        efetch_response = session.get(EFETCH_URL, params=efetch_params)
        if not self._uses_cache or not efetch_response.from_cache:
            # throttle if not using cache or if not cached
            logger.debug(f"Sleeping for {RATE_LIMIT_DELAY} seconds")
            time.sleep(RATE_LIMIT_DELAY)
        if not efetch_response.ok:
            logger.error(f"Failed to fetch data for {pubmed_ids}")
            raise ValueError(
                f"Failed to fetch data for {pubmed_ids} using {session} and {efetch_params}"
            )
        medline_records = efetch_response.text

        # Parsing titles and abstracts from the MEDLINE records
        parsed_data = []
        current_record = {}
        current_field = None

        for line in medline_records.split("\n"):
            if line.startswith("PMID- "):
                current_field = "id"
                current_record[current_field] = "PMID:" + line.replace("PMID- ", "").strip()
            if line.startswith("PMC - "):
                current_field = "pmcid"
                current_record[current_field] = "PMCID:" + line.replace("PMC - ", "").strip()
            elif line.startswith("TI  - "):
                current_field = "title"
                current_record[current_field] = line.replace("TI  - ", "").strip()
            elif line.startswith("AB  - "):
                current_field = "abstract"
                current_record[current_field] = line.replace("AB  - ", "").strip()
            elif line.startswith("    "):  # Continuation of the previous field
                if current_field and current_field in current_record:
                    current_record[current_field] += " " + line.strip()
            else:
                current_field = None

            if line == "":
                if current_record:
                    parsed_data.append(current_record)
                    current_record = {}
        return parsed_data
