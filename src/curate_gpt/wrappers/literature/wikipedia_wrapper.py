"""Chat with a KB."""

import logging
import time
from dataclasses import dataclass
from typing import ClassVar, List

import inflection
import requests

from curate_gpt.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)

BASE_URL = "https://en.wikipedia.org/w/api.php"


@dataclass
class WikipediaWrapper(BaseWrapper):
    """
    A wrapper to provide a search facade over Wikipedia.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
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
        response = requests.get(BASE_URL, params=params)
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

        info_response = requests.get(BASE_URL, params=info_params)
        if not info_response.ok:
            raise ValueError(f"Could not get info for {titles}")
        info_data = info_response.json()

        # Extracting and printing the information
        if "query" not in info_data:
            logger.error(f"Could not get pages from {info_data}")
            return []
        pages = info_data["query"]["pages"]
        results = []
        for _page_id, page_info in pages.items():
            page_info["id"] = inflection.camelize(page_info["title"])
            page_info["snippet"] = snippets[page_info["title"]]
            del page_info["ns"]
            del page_info["pageid"]
            results.append(page_info)
        return results
