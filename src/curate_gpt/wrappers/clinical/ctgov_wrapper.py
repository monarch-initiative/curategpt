"""Chat with a KB."""

import logging
import time
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional

import requests
import requests_cache
from oaklib import BasicOntologyInterface

from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2"
STUDY_URL = f"{BASE_URL}/studies"

RATE_LIMIT_DELAY = 0.5


@dataclass
class ClinicalTrialsWrapper(BaseWrapper):
    """
    A wrapper over a clinicaltrials.gov.

    """

    name: ClassVar[str] = "ctgov"

    _label_adapter: BasicOntologyInterface = None

    default_object_type = "ClinicalTrial"

    session: requests.Session = field(default_factory=lambda: requests.Session())

    where: Optional[Dict] = None

    _uses_cache: bool = False

    def set_cache(self, name: str) -> None:
        self.session = requests_cache.CachedSession(name)
        self._uses_cache = True

    def external_search(
        self, text: str, expand: bool = True, where: Optional[Dict] = None, **kwargs
    ) -> List:
        if expand:
            logger.info(f"Expanding search term: {text} to create ctgov query")
            model = self.extractor.model
            response = model.prompt(
                text,
                system=(
                    "generate a MINIMAL semi-colon separated list of the most relevant terms. "
                    "make terms general such that when they are concatenated with AND they can be "
                    "used to search for the desired clinical trial."
                ),
            )
            terms = response.text().split(";")
            terms = [f'"{term}"' for term in terms]
            search_term = " AND ".join(terms)
        else:
            search_term = text
        if not where and self.where:
            where = self.where
        if where:
            whereq = " AND ".join([f"{v}[{k}]" for k, v in where.items()])
            search_term = f"({search_term}) AND {whereq}"
        logger.info(f"Constructed search term: {search_term}")
        # Parameters for the request
        params = {
            "query.term": search_term,
        }
        # Note: we don't cache this call as there could be many
        # different search terms
        response = requests.get(STUDY_URL, params=params)
        time.sleep(RATE_LIMIT_DELAY)
        data = response.json()
        return self.objects_from_list(data["studies"])

    def objects_from_list(self, input_objs: List[Dict]) -> List[Dict]:
        objs = []
        for input_obj in input_objs:
            protocolSection = input_obj["protocolSection"]
            identificationModule = protocolSection["identificationModule"]
            # print(protocolSection)
            obj = {
                "id": identificationModule["nctId"],
                "title": identificationModule["briefTitle"],
                "description": protocolSection["descriptionModule"],
            }
            objs.append(obj)
        return objs
