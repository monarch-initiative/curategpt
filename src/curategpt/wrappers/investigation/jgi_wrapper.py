"""Chat with a KB."""

import logging
import time
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional

import requests

from curate_gpt.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)

BASE_URL = "https://files.jgi.doe.gov/search/"


@dataclass
class JGIWrapper(BaseWrapper):
    """
    A wrapper to provide a search facade over JGI Data Search.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
    """

    name: ClassVar[str] = "jgi"

    def external_search(
        self, text: str, expand: bool = False, where: Optional[Dict] = None, **kwargs
    ) -> List:
        params = {
            "x": 20,
        }
        if expand:

            def qt(t: str):
                t = t.strip()
                if " " in t:
                    return f'"{t}"'
                return t

            logger.info(f"Expanding search term: {text} to create JGI query")
            model = self.extractor.model
            response = model.prompt(
                text,
                system="""
                Take the specified search text, and expand it to a list
                of key terms used to construct a query. You will return results as
                semi-colon separated list of the most relevant terms. Make sure to
                include all relevant concepts in the returned terms.""",
            )
            terms = response.text().split(";")
            logger.info(f"Expanded terms: {terms}")
            terms = [qt(t) for t in terms]
            # terms = terms[0:2]
            search_term = "|".join(terms)
        else:
            search_term = text
        params["q"] = search_term
        if where:
            params.update(where)
        time.sleep(0.5)
        logger.info(f"Constructed query: {params}")
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        organisms = data["organisms"]
        logger.info(f"Got {len(organisms)} organisms")
        results = []
        fields = [
            "doi",
            "name",
            "label",
            "title",
            "product_search_category",
            "status",
        ]
        for org in organisms:
            result = {}
            for k in fields:
                if k in org:
                    result[k] = org[k]
            files = org.get("files", [])
            if files:
                top_file = files[0]
                md = top_file.get("metadata", {})
                gold = md.get("gold", {})
                if gold:
                    result["display_name"] = gold.get("display_name", None)
                    result["ecosystem"] = gold.get("ecosystem", None)
            results.append(result)
        return results
