"""Chat with a KB."""
import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, List

from oaklib import BasicOntologyInterface
from pytrials.client import ClinicalTrials

from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)


@dataclass
class ClinicalTrialsWrapper(BaseWrapper):
    """
    A wrapper over a clinicaltrials.gov.

    """

    name: ClassVar[str] = "ctgov"

    _label_adapter: BasicOntologyInterface = None

    default_object_type = "ClinicalTrial"

    def external_search(self, text: str, expand: bool = True, **kwargs) -> List[Dict]:
        ct = ClinicalTrials()
        logger.info(ct)
        # TODO
        return []
