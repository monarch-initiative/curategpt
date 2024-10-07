"""Chat with a KB."""

import logging
import time
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional

import requests

from curategpt.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)

BASE_URL = "http://wwwdev.ebi.ac.uk/Tools/omicsdi/ws"


@dataclass
class OmicsDIWrapper(BaseWrapper):
    """
    A wrapper to provide a search facade over OMICS DI Search.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
    """

    name: ClassVar[str] = "omicsdi"

    default_object_type = "Dataset"

    source: str = None  # pride, ...

    def external_search(
        self, text: str, expand: bool = False, where: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        """
        Search the OmicsDI database for the given text.

        TODO: full text retrieval doesn't seem to work well
        """
        params = {
            "size": 20,
            "faceCount": 30,
        }
        if expand:

            def qt(t: str):
                t = t.strip()
                if " " in t:
                    return f'"{t}"'
                return t

            logger.info(f"Expanding search term: {text} to create OmicsDI query")
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
            search_term = " ".join(terms)
        else:
            search_term = text
        params["query"] = search_term
        if where:
            params.update(where)
        time.sleep(0.25)
        logger.info(f"Constructed query: {params}")
        url = f"{BASE_URL}/dataset/search"
        logger.info(f"Searching OmicsDI {url} with query: {params}")
        response = requests.get(url, params=params)
        data = response.json()

        datasets = data["datasets"]
        logger.info(f"Found {len(datasets)} datasets")
        for dataset in datasets:
            dataset["additional"] = self.additional_info(dataset["id"], dataset["source"])
        return datasets

    def additional_info(self, local_id: str, source: str):
        """
        Augment the local ID with information from the database.
        """
        url = f"{BASE_URL}/dataset/{source}/{local_id}"
        logger.info(f"Getting additional info from {url}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data["additional"]

    def objects_by_ids(self, ids: List[str], source: str = None):
        """
        Augment the local ID with information from the database.
        """
        source = source or self.source
        data_objects = []
        for id in ids:
            if ":" in id:
                source, local_id = id.split(":")
            else:
                local_id = id
            if not source:
                raise ValueError(f"Need a source for ID {id}")
            source = source.lower()
            url = f"{BASE_URL}/dataset/{source}/{local_id}"
            logger.info(f"Getting additional info from {url}")
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            data_objects.append(data)

        return data_objects
