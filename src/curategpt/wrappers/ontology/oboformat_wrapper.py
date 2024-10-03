"""Chat with a KB."""

import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, Iterator, Optional

from curategpt.wrappers.base_wrapper import BaseWrapper

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
