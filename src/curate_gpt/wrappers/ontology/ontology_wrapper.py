"""Chat with a KB."""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Iterable, Iterator, List, Mapping, Optional

from oaklib import BasicOntologyInterface
from oaklib.datamodels.search import SearchConfiguration
from oaklib.datamodels.vocabulary import IS_A
from oaklib.interfaces import OboGraphInterface, SearchInterface
from oaklib.types import CURIE
from oaklib.utilities.iterator_utils import chunk

from curate_gpt.formatters.format_utils import camelify
from curate_gpt.wrappers.base_wrapper import BaseWrapper
from curate_gpt.wrappers.ontology.ontology import OntologyClass, Relationship

logger = logging.getLogger(__name__)


@dataclass
class OntologyWrapper(BaseWrapper):
    """
    A wrapper to pull from ontologies using OAK.

    This wrapper can be used either with static sources (e.g. an ontology file)
    or dynamic (pubmed)
    """

    name: ClassVar[str] = "oaklib"

    oak_adapter: BasicOntologyInterface = None

    id_to_shorthand: Mapping[CURIE, CURIE] = None
    shorthand_to_id: Mapping[CURIE, CURIE] = None
    _objects_by_curie: Mapping[CURIE, OntologyClass] = None
    _objects_by_shorthand: Mapping[str, OntologyClass] = None

    default_max_search_results: int = 500
    fetch_definitions: bool = field(default=True)
    fetch_relationships: bool = field(default=True)

    def __post_init__(self):
        """Initialize."""
        self.id_to_shorthand = {}

    def external_search(self, text: str, expand: bool = True, limit: int = None, **kwargs) -> List:
        adapter = self.oak_adapter
        if not isinstance(adapter, SearchInterface):
            raise ValueError(f"OAK adapter {self.oak_adapter} does not support search")
        cfg = SearchConfiguration(is_partial=True)
        chunk_iter = chunk(
            adapter.basic_search(text, cfg), limit or self.default_max_search_results
        )
        for object_ids in chunk_iter:
            print("object_ids", object_ids)
            return list(self.objects(object_ids=object_ids))

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        """
        Yield all objects in the view.

        :return:
        """
        adapter = self.oak_adapter
        entities = list(object_ids if object_ids else adapter.entities())
        labels = {e: lbl for e, lbl in adapter.labels(entities, allow_none=False)}
        definitions = {}
        if self.fetch_definitions:
            for chunked_entities in chunk(entities, 100):
                for id, defn, _ in adapter.definitions(chunked_entities):
                    definitions[id] = defn
        relationships = defaultdict(list)
        if self.fetch_relationships:
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
            yield obj.dict()

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
