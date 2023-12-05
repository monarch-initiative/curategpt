"""Chat with a KB."""
import gzip
import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, Iterator, Optional

from oaklib import BasicOntologyInterface, get_adapter

from curate_gpt.formatters.format_utils import camelify
from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)


RMAP = {"CID": "induces"}


@dataclass
class BiocWrapper(BaseWrapper):

    """
    A wrapper over a bioc source file.

    This provides an exhaustive list of objects.
    """

    name: ClassVar[str] = "bioc"

    _label_adapter: BasicOntologyInterface = None

    default_object_type = "AnnotatedPublication"

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        from bioc import biocxml

        with gzip.open(str(self.source_locator), "rb") as f:
            collection = biocxml.load(f)
            for document in collection.documents:
                doc = {}
                for p in document.passages:
                    # title and abstract are the only two passages
                    doc[p.infons["type"]] = p.text
                rels = []
                for r in document.relations:
                    obj = {}
                    for k, v in r.infons.items():
                        if k == "relation":
                            obj["relation"] = RMAP.get(v, v)
                        else:
                            mesh_id = f"MESH:{v}"
                            label = self.label_adapter.label(mesh_id)
                            if not label:
                                logger.warning(f"Could not find label for {mesh_id}")
                            obj[k] = camelify(label)
                    rels.append(obj)
                doc["statements"] = rels
                yield doc

    @property
    def label_adapter(self) -> BasicOntologyInterface:
        """Get the label adapter."""
        if self._label_adapter is None:
            self._label_adapter = get_adapter("sqlite:obo:mesh")
        return self._label_adapter
