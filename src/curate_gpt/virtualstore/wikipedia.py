"""Chat with a KB."""
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterator, List, Optional

import inflection
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

SEARCH_URL = "https://en.wikipedia.org/w/api.php"


@dataclass
class WikipediaView(DBView):
    """
    An agent to pull from wikipedia.
    """

    name: ClassVar[str] = "wikipedia"

    def external_search(self, text: str, expand: bool = False, **kwargs) -> List:
        # if expand:
        #    raise NotImplementedError
        # else:
        #    search_term = text
        search_term = text
        logger.info(f"Constructed search term: {search_term}")
        # Parameters for the request
        params = {"action": "query", "format": "json", "list": "search", "srsearch": search_term}

        time.sleep(0.5)
        response = requests.get(SEARCH_URL, params=params)
        data = response.json()
        search_results = data["query"]["search"]
        snippets = {result["title"]: result["snippet"] for result in search_results}
        titles = list(snippets.keys())

        info_params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(titles),
            "prop": "extracts",
            "exintro": True,  # Get only the introduction of each page
            "explaintext": True,  # Get plain text instead of HTML
        }

        info_response = requests.get(SEARCH_URL, params=info_params)
        info_data = info_response.json()

        # Extracting and printing the information
        pages = info_data["query"]["pages"]
        results = []
        for page_id, page_info in pages.items():
            page_info["id"] = inflection.camelize(page_info["title"])
            page_info["snippet"] = snippets[page_info["title"]]
            del page_info["ns"]
            del page_info["pageid"]
            results.append(page_info)
        return results
