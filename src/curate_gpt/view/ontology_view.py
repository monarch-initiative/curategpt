"""A View that bridges to an OAK ontology."""
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Union

import inflection
from oaklib import BasicOntologyInterface
from oaklib.datamodels.vocabulary import IS_A
from oaklib.interfaces import OboGraphInterface
from oaklib.types import CURIE
from oaklib.utilities.iterator_utils import chunk
from pydantic import BaseModel

from curate_gpt.view.ontology import Ontology, OntologyClass, Relationship
from curate_gpt.view.view import View

logger = logging.getLogger(__name__)


def camelify(text: str) -> str:
    """
    Convert text to camel case.

    :param text:
    :return:
    """
    # replace all non-alphanumeric characters with underscores
    safe = "".join([c if c.isalnum() else "_" for c in text])
    return inflection.camelize(safe)


@dataclass
class OntologyView(View):
    """
    A view of an ontology.

    Uses OAK.
    """

    adapter: BasicOntologyInterface = None
    id_to_shorthand: Mapping[CURIE, CURIE] = None
    shorthand_to_id: Mapping[CURIE, CURIE] = None
    _objects_by_curie: Mapping[CURIE, OntologyClass] = None
    _objects_by_shorthand: Mapping[str, OntologyClass] = None

    def __post_init__(self):
        """Initialize."""
        self.id_to_shorthand = {}

    def objects(self) -> Iterator[OntologyClass]:
        """
        Yield all objects in the view.

        :return:
        """
        adapter = self.adapter
        entities = list(adapter.entities())
        labels = {e: lbl for e, lbl in adapter.labels(entities, allow_none=False)}
        definitions = {}
        for chunked_entities in chunk(entities, 100):
            for id, defn, _ in adapter.definitions(chunked_entities):
                definitions[id] = defn
        relationships = defaultdict(list)
        for sub, pred, obj in adapter.relationships():
            relationships[sub].append((pred, obj))
        self.id_to_shorthand = {}
        self.shorthand_to_id = {}
        for id, lbl in labels.items():
            shorthand = camelify(lbl)
            if shorthand in self.shorthand_to_id:
                shorthand = f"{shorthand}_{id}"
            if shorthand in self.shorthand_to_id:
                continue
            self.id_to_shorthand[id] = shorthand
            self.shorthand_to_id[shorthand] = id
        self._objects_by_curie = {}
        self._objects_by_shorthand = {}
        for id, shorthand in self.id_to_shorthand.items():
            obj = OntologyClass(
                id=shorthand,
                label=labels[id],
                relationships=[
                    Relationship(predicate=self._as_shorthand(pred), target=self._as_shorthand(obj))
                    for pred, obj in relationships.get(id, [])
                ],
            )
            if id in definitions:
                obj.definition = definitions[id]
            self._objects_by_curie[id] = obj
        # for id, alias in adapter.alias_relationships():
        if isinstance(adapter, OboGraphInterface):
            for ldef in adapter.logical_definitions():
                shorthand = self._as_shorthand(ldef.definedClassId)
                obj = self._objects_by_curie.get(shorthand, None)
                if obj is None:
                    continue
                obj.logical_definition = [
                    Relationship(predicate=IS_A, target=self._as_shorthand(obj))
                    for pred, obj in ldef.genusIds
                ] + [
                    Relationship(
                        predicate=self._as_shorthand(r.propertyId),
                        target=self._as_shorthand(r.fillerId),
                    )
                    for r in ldef.restrictions
                ]
        for obj in self._objects_by_curie.values():
            yield obj

    def as_object(self, curie: CURIE) -> Optional[OntologyClass]:
        if not self._objects_by_curie:
            list(self.objects())
        return self._objects_by_curie.get(curie, None)

    def _as_shorthand(self, curie: CURIE) -> CURIE:
        return self.id_to_shorthand.get(curie, curie)

    @property
    def text_field(self) -> Callable:
        """
        Returns a function that returns the embeddable text field for an object.

        :return:
        """
        # TODO: decouple from OntologyClass
        return lambda obj: obj.label if isinstance(obj, OntologyClass) else obj.get("label")
        # return lambda obj: f"{obj.label}: {obj.definition}" if isinstance(obj, OntologyClass) else f'{obj.get("label")}: {obj.get("definition", "")}'
