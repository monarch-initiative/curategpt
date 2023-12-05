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
class OntologyWrapper(BaseWrapper):

    """
    A wrapper to pull from ontologies using OAK.

    This wrapper can be used either with static sources (e.g. an ontology file)
    or dynamic (pubmed).

    The data model is a simple JSON/Dict structure
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
    relationships_as_fields: bool = field(default=False)

    branches: List[str] = None

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
        entities = list(adapter.entities())

        if self.branches:
            if not isinstance(adapter, OboGraphInterface):
                raise ValueError(f"OAK adapter {self.oak_adapter} does not support branches")
            selected_ids = []
            for branch in self.branches:
                selected_ids.extend(list(adapter.descendants([branch], predicates=[IS_A])))
            selected_ids = list(set(selected_ids))
        elif object_ids:
            selected_ids = list(object_ids)
        else:
            selected_ids = list(entities)
        logger.info(f"Found {len(selected_ids)} selected ids")
        # need to fetch ALL labels in store, even if not selected,
        # as may be used in references
        labels = {e: lbl for e, lbl in adapter.labels(entities, allow_none=False)}
        logger.info(f"Found {len(labels)} labels")
        definitions = {}
        if self.fetch_definitions:
            for chunked_entities in chunk(selected_ids, 100):
                for id, defn, _ in adapter.definitions(chunked_entities):
                    definitions[id] = defn
        relationships = defaultdict(list)
        if self.fetch_relationships:
            for sub, pred, obj in adapter.relationships():
                if sub in entities:
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
            if id not in selected_ids:
                continue
            obj = OntologyClass(
                id=shorthand,
                label=labels[id],
                original_id=id,
            )
            for pred, tgt in relationships.get(id, []):
                k = self._as_shorthand(pred)
                k = k.replace("rdfs:", "")
                if self.relationships_as_fields:
                    if not hasattr(obj, k):
                        setattr(obj, k, [])
                    getattr(obj, k).append(self._as_shorthand(tgt))
                else:
                    if not obj.relationships:
                        obj.relationships = []
                    obj.relationships.append(
                        Relationship(predicate=k, target=self._as_shorthand(tgt))
                    )
            if id in definitions:
                obj.definition = definitions[id]
            self._objects_by_curie[id] = obj
        # for id, alias in adapter.alias_relationships():
        if isinstance(adapter, OboGraphInterface):
            for ldef in adapter.logical_definitions(adapter.entities()):
                shorthand = self._as_shorthand(ldef.definedClassId)
                obj = self._objects_by_curie.get(ldef.definedClassId, None)
                if obj is None:
                    continue
                obj.logical_definition = [
                    Relationship(predicate=IS_A, target=self._as_shorthand(g))
                    for g in ldef.genusIds
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

    def retrieve_shorthand_to_id_from_store(self, store: DBAdapter) -> Mapping[str, str]:
        if not self.shorthand_to_id:
            self.shorthand_to_id = {}
            for obj, _, __ in store.find({}):
                self.shorthand_to_id[obj["id"]] = obj["original_id"]
        return self.shorthand_to_id

    def unwrap_object(self, obj: Dict[str, Any], store: DBAdapter, **kwargs) -> og.Graph:
        return self.unwrap_objects([obj], store, **kwargs)

    def unwrap_objects(
        self, objs: Iterable[Dict[str, Any]], store: DBAdapter, drop_dangling=False, **kwargs
    ) -> og.GraphDocument:
        """
        Convert an object from the store to the view representation.

        reverse transform of `as_object`

        :param object:
        :param kwargs:
        :return:
        """
        m = self.retrieve_shorthand_to_id_from_store(store)
        graph = og.Graph(id="tmp", nodes=[], edges=[])
        for obj in objs:
            if not obj:
                logger.warning(f"Skipping empty object {obj}")
                continue
            id = obj.get("original_id", None)
            if not id:
                logger.warning(f"Skipping empty id in {obj}")
                continue
            node = og.Node(
                id=id,
                lbl=obj.get("label", None),
                type="CLASS",
            )
            meta = {}
            defn = obj.get("definition", None)
            if defn:
                meta["definition"] = {"val": defn}
            node.meta = meta
            graph.nodes.append(node)
            for rel in obj.get("relationships", []):
                tgt = rel.get("target", None)
                if not tgt:
                    logger.warning(f"Skipping empty target in {rel}")
                    continue
                if isinstance(tgt, list):
                    logger.warning(f"Unexpected list: {tgt}")
                    tgt = tgt[0]
                if tgt in m:
                    tgt = m[tgt]
                else:
                    if drop_dangling:
                        logger.warning(f"Could not find {tgt}")
                        continue
                    else:
                        tgt = tgt.replace(":", "_")
                pred = rel["predicate"]
                if pred in m:
                    pred = m[pred]
                else:
                    logger.warning(f"Could not find {pred}")
                if pred == "subClassOf":
                    pred = "is_a"
                edge = og.Edge(
                    sub=id,
                    obj=tgt,
                    pred=pred,
                )
                graph.edges.append(edge)
        return og.GraphDocument(graphs=[graph])
