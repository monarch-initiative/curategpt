"""Chat with a KB."""
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, ClassVar, Optional, Iterator

import requests
import yaml
from eutils import Client
from pydantic import BaseModel

from curate_gpt.agents.chat import ChatEngine, ChatResponse
from curate_gpt.extract import AnnotatedObject, Extractor
from curate_gpt.store import DBAdapter
from curate_gpt.store.db_adapter import SEARCH_RESULT

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

PUBMED_COLLECTION_NAME = "pubmed_subset"
PUBMED_TEMP_COLLECTION_NAME = "__pubmed_temp__"
PUBMED_EMBEDDING_MODEL = "openai:"


@dataclass
class PubmedAgent:
    """
    An agent to pull from pubmed.

    TODO: make this a virtual store
    """

    local_store: DBAdapter = None
    """Adapter to local knowledge base"""

    eutils_client: Client = None

    extractor: Extractor = None

    def search(
        self,
        text: str,
        collection: str = None,
        cache: bool = True,
        expand: bool = True,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        """
        Extract structured object using text seed and background knowledge.

        :param text:
        :param kwargs:
        :return:
        """
        if collection is None:
            collection = PUBMED_COLLECTION_NAME
        logger.info(f"Searching for {text}, caching in {collection}")
        if expand:
            logger.info(f"Expanding search term: {text} to create pubmed query")
            model = self.extractor.model
            response = model.prompt(text, system="generate a semi-colon separated list of the most relevant terms")
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
            "retmode": "json"
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
                return self.search(text, collection=collection, cache=cache, expand=False, **kwargs)
            else:
                logger.error(f"Failed to find results for {text}")
                return

        logger.info(f"Found {len(pubmed_ids)} results: {pubmed_ids}")

        efetch_params = {
            "db": "pubmed",
            "id": ",".join(pubmed_ids),  # Combine PubMed IDs into a comma-separated string
            "retmode": "json"
        }

        # Parameters for the efetch request
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(pubmed_ids),  # Combine PubMed IDs into a comma-separated string
            "rettype": "medline",
            "retmode": "text"
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
        db = self.local_store
        if not cache:
            collection = PUBMED_TEMP_COLLECTION_NAME
            db.remove_collection(collection, exists_ok=True)
        logger.info(f"Inserting {len(parsed_data)} records into {collection}")
        db.upsert(parsed_data, collection=collection, model=PUBMED_EMBEDDING_MODEL)
        db.update_collection_metadata(collection, description=f"Special cache for pubmed searches")
        yield from db.search(text, collection=collection, **kwargs)

    def chat(
            self,
            query: str,
            collection: str = None,
            **kwargs,
    ) -> ChatResponse:
        """
        Chat with pubmed.

        :param query:
        :param collection:
        :param kwargs:
        :return:
        """
        # prime the pubmed cache
        if collection is None:
            collection = PUBMED_COLLECTION_NAME
        logger.info(f"Searching pubmed for {query}, kwargs={kwargs}, self={self}")
        self.search(query, collection=collection, **kwargs)
        chat = ChatEngine(kb_adapter=self.local_store, extractor=self.extractor)
        response = chat.chat(query, collection=collection)
        return response






