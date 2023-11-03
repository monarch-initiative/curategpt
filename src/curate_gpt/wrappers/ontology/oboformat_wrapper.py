"""Chat with a KB."""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterable, Iterator, List, Mapping, Optional

import oaklib.datamodels.obograph as og
from oaklib import BasicOntologyInterface
from oaklib.datamodels.search import SearchConfiguration
from oaklib.datamodels.vocabulary import IS_A
from oaklib.interfaces import OboGraphInterface, SearchInterface
from oaklib.types import CURIE
from oaklib.utilities.iterator_utils import chunk

from curate_gpt import DBAdapter
from curate_gpt.formatters.format_utils import camelify
from curate_gpt.wrappers.base_wrapper import BaseWrapper
from curate_gpt.wrappers.ontology.ontology import OntologyClass, Relationship

logger = logging.getLogger(__name__)


@dataclass
class OBOFormatWrapper(BaseWrapper):
    """
    A wrapper to index ontologies in OBO Format.

    Note that in contrast to the OntologyWrapper, this preserves OBO syntax stanzas, e.g.

    ```json
    {"term": "[Term]\nid: ...\nname: ...\n..."}
    ```
    """

    name: ClassVar[str] = "oboformat"

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        """
        Yield all objects in the view.

        :return:
        """

        path = self.source_locator

        with open(path) as file:
            obj = None
            for line in file.readlines():
                if line.startswith("["):
                    yield from self.wrap_object(obj)
                    obj = ""
                if obj is not None:
                    obj += line
            yield from self.wrap_object(obj)

    def wrap_object(self, obj: Optional[str]) -> Dict:
        if obj:
            yield {"term": obj}
