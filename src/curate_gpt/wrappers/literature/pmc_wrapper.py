"""EUtils-based wrapper for studies in NCBI."""

import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, List

import yaml

from curate_gpt.wrappers.literature.eutils_wrapper import EUtilsWrapper

logger = logging.getLogger(__name__)


@dataclass
class PMCWrapper(EUtilsWrapper):
    """
    A wrapper to provide a search facade over PMC via eutils.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
    """

    name: ClassVar[str] = "pmc"

    eutils_db: ClassVar[str] = "pmc"

    id_prefix: ClassVar[str] = "pmc"

    default_object_type = "Reference"

    fetch_tool = "esummary"

    def objects_from_dict(self, results: Dict) -> List[Dict]:
        print(yaml.dump(results))
        raise NotImplementedError()

        objs = []
        # TODO
        # _articles = results["pmc-articleset"]["article"]
        return objs
