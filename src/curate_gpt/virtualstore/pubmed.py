"""Chat with a KB."""
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterator, List, Optional

import requests
import yaml
from eutils import Client
from pydantic import BaseModel

from curate_gpt.agents.chat import ChatEngine, ChatResponse
from curate_gpt.extract import AnnotatedObject, Extractor
from curate_gpt.store import DBAdapter
from curate_gpt.store.db_adapter import SEARCH_RESULT
from curate_gpt.virtualstore.dbview import DBView

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

PUBMED_COLLECTION_NAME = "pubmed_api_cached"
PUBMED_TEMP_COLLECTION_NAME = "__pubmed_temp__"
PUBMED_EMBEDDING_MODEL = "openai:"


@dataclass
class PubmedView(DBView):
    """
    An agent to pull from pubmed.
    """

    name: ClassVar[str] = "pubmed"

    eutils_client: Client = None

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

        time.sleep(0.5)
        response = requests.get(ESEARCH_URL, params=params)
        data = response.json()

        # Extract PubMed IDs from the response
        pubmed_ids = data["esearchresult"]["idlist"]

        if not pubmed_ids:
            logger.warning(f"No results with {search_term}")
            if expand:
                logger.info(f"Trying again without expansion")
                return self.external_search(text, expand=False, **kwargs)
            else:
                logger.error(f"Failed to find results for {text}")
                return []

        logger.info(f"Found {len(pubmed_ids)} results: {pubmed_ids}")

        efetch_params = {
            "db": "pubmed",
            "id": ",".join(pubmed_ids),  # Combine PubMed IDs into a comma-separated string
            "retmode": "json",
        }

        # Parameters for the efetch request
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(pubmed_ids),  # Combine PubMed IDs into a comma-separated string
            "rettype": "medline",
            "retmode": "text",
        }
        efetch_response = requests.get(EFETCH_URL, params=efetch_params)
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
