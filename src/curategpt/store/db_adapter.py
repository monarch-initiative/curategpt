"""Abstract DB adapter."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Iterable, Iterator, List, Optional, TextIO, Union, Tuple

import pandas as pd
import yaml
from click.utils import LazyFile
from jsonlines import jsonlines

from curategpt.store.metadata import Metadata
from curategpt.store.schema_proxy import SchemaProxy
from curategpt.store.vocab import (
    DEFAULT_COLLECTION,
    DOCUMENTS,
    EMBEDDINGS,
    FILE_LIKE,
    METADATAS,
    OBJECT,
    PROJECTION,
    QUERY,
    SEARCH_RESULT,
)

logger = logging.getLogger(__name__)


def _get_file(file: Optional[FILE_LIKE] = None, mode="r") -> Optional[TextIO]:
    if file is None:
        return None
    if isinstance(file, Path):
        file = str(file)
    if isinstance(file, str):
        return open(file, mode)
    elif isinstance(file, TextIO):
        return file
    elif isinstance(file, LazyFile):
        return file
    else:
        raise TypeError(f"Unknown file type: {type(file)}")


@dataclass
class DBAdapter(ABC):
    """
    Base class for stores.

    This base class provides a common interface for a wide variety of document or object stores.
    The interface is intended to closely mimic the kind of interface found for document stores
    such as mongoDB or vector databases such as ChromaDB, but the intention is that can be
    used for SQL databases, SPARQL endpoints, or even file systems.

    The store allows for storage and retrieval of *objects* which are arbitrary dictionary
    objects, equivalient to a JSON object.

    Objects are partitioned into *collections*, which maps to the equivalent concept in
    MongoDB and ChromaDB.

    >>> from curategpt.store import get_store
    >>> store = get_store("in_memory")
    >>> store.insert({"name": "John", "age": 42}, collection="people")

    If you are used to working with MongoDB and ChromaDB APIs directly, one difference is that
    here we do not provide a separate Collection object, everything is handled through the
    store object. You can optionally bind a store object to a collection, which effectively
    gives you a collection object:

    >>> from curategpt.store import get_store
    >>> store = get_store("in_memory")
    >>> store.set_collection("people")
    >>> store.insert({"name": "John", "age": 42})

    TODO: decide if this is the final interface
    """

    path: str = None
    """Path to a location where the database is stored or disk or the network."""

    name: ClassVar[str] = "base"

    schema_proxy: Optional[SchemaProxy] = None
    """Schema manager"""

    collection: Optional[str] = None
    """Default collection"""

    index_fields: Optional[Union[List[str], Tuple[str]]] = None
    """Fields containing text to embed"""

    # _field_names_by_collection: Dict[str, Optional[List[str]]] = field(default_factory=dict)
    _field_names_by_collection: Dict[str, Optional[List[str]]] = None

    # CUD operations

    @abstractmethod
    def insert(self, objs: Union[OBJECT, Iterable[OBJECT]], collection: str = None, **kwargs):
        """
        Insert an object or list of objects into the store.

        >>> from curategpt.store import get_store
        >>> store = get_store("in_memory")
        >>> store.insert([{"name": "John", "age": 42}], collection="people")

        :param objs:
        :param collection:
        :return:
        """

    def update(self, objs: Union[OBJECT, List[OBJECT]], collection: str = None, **kwargs):
        """
        Update an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        raise NotImplementedError

    def upsert(self, objs: Union[OBJECT, List[OBJECT]], collection: str = None, **kwargs):
        """
        Upsert an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        raise NotImplementedError

    def delete(self, id: str, collection: str = None, **kwargs):
        """
        Delete an object by its ID.

        :param id:
        :param collection:
        :return:
        """
        raise NotImplementedError

    # View operations

    def create_view(self, view_name: str, collection: str, expression: QUERY, **kwargs):
        """
        Create a view in the database.

        Todo:
        ----
        :param view:
        :return:
        """
        raise NotImplementedError

    # Collection operations

    def set_collection(self, collection: str):
        """
        Set the current collection.

        If this is set, then all subsequent operations will be performed on this collection, unless
        overridden.

        This allows the following

        >>> from curategpt.store import get_store
        >>> store = get_store("in_memory")
        >>> store.set_collection("people")
        >>> store.insert([{"name": "John", "age": 42}])

        to be written in place of

        >>> from curategpt.store import get_store
        >>> store = get_store("in_memory")
        >>> store.insert([{"name": "John", "age": 42}], collection="people")

        :param collection:
        :return:
        """
        self.collection = collection

    def _get_collection(self, collection: Optional[str] = None):
        if collection is not None:
            return collection
        elif self.collection is not None:
            return self.collection
        else:
            return DEFAULT_COLLECTION

    def remove_collection(self, collection: str = None, exists_ok=False, **kwargs):
        """
        Remove a collection from the database.

        :param collection:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def list_collection_names(self) -> List[str]:
        """
        List all collections in the database.

        :return: names of collections
        """

    @abstractmethod
    def collection_metadata(
        self, collection_name: Optional[str] = None, include_derived=False, **kwargs
    ) -> Optional[Metadata]:
        """
        Get the metadata for a collection.

        :param collection_name:
        :param include_derived: Include derived metadata, e.g. counts
        :return:
        """

    def set_collection_metadata(
        self, collection_name: Optional[str], metadata: Metadata, **kwargs
    ):
        """
        Set the metadata for a collection.

        >>> from curategpt.store import get_store
        >>> from curategpt.store import Metadata
        >>> store = get_store("in_memory")
        >>> md = store.collection_metadata(collection)
        >>> md.venomx.id == "People"
        >>> md.venomx.embedding_model.name == "openai:"
        >>> store.set_collection_metadata("people", cm)

        :param collection_name:
        :return:
        """
        raise NotImplementedError

    def update_collection_metadata(self, collection_name: str, **kwargs) -> Metadata:
        """
        Update the metadata for a collection.

        :param collection_name:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self, text: str, where: QUERY = None, collection: str = None, **kwargs
    ) -> Iterator[SEARCH_RESULT]:
        """
        Query the database for a text string.

        >>> from curategpt.store import get_store
        >>> store = get_store("chromadb", "db")
        >>> for obj, distance, info in store.search("forebrain neurons", collection="ont_cl"):
        ...     obj_id = obj["id"]
        ...     # print at precision of 2 decimal places
        ...     print(f"{obj_id} {distance:.2f}")
        <BLANKLINE>
        ...
        NeuronOfTheForebrain 0.28
        ...

        :param text:
        :param collection:
        :param where:
        :param kwargs:
        :return: tuple of object, distance, metadata
        """

    def find(
        self,
        where: QUERY = None,
        projection: PROJECTION = None,
        collection: str = None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        """
        Query the database.

        >>> from curategpt.store import get_store
        >>> store = get_store("chromadb", "db")
        >>> objs = list(store.find({"name": "NeuronOfTheForebrain"}, collection="ont_cl"))

        :param collection:
        :param where:
        :param projection:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def matches(self, obj: OBJECT, **kwargs) -> Iterator[SEARCH_RESULT]:
        """
        Query the database for matches to an object.

        :param obj:
        :param kwargs:
        :return:
        """

    @abstractmethod
    def lookup(self, id: str, collection: str = None, **kwargs) -> OBJECT:
        """
        Lookup an object by its ID.

        :param id:
        :param collection:
        :return:
        """

    def lookup_multiple(self, ids: List[str], **kwargs) -> Iterator[OBJECT]:
        """
        Lookup an object by its ID.

        :param id:
        :param collection:
        :return:
        """
        yield from [self.lookup(id, **kwargs) for id in ids]

    @abstractmethod
    def peek(self, collection: str = None, limit=5, **kwargs) -> Iterator[OBJECT]:
        """
        Peek at first N objects in a collection.

        :param collection:
        :param limit:
        :return:
        """
        raise NotImplementedError

    # Schema operations

    @abstractmethod
    def fetch_all_objects_memory_safe(
        self, collection: str = None, batch_size: int = 100, **kwargs
    ) -> Iterator[OBJECT]:
        """
        Fetch all objects from a collection, in batches to avoid memory overload.
        """
        raise NotImplementedError

    def text_to_embed_or_lookup(self, obj: Dict):
        vals = [str(obj.get(f) for f in self.index_fields if f in obj)]
        return " ".join(vals)


    def identifier_field(self, collection: str = None) -> str:
        # TODO: use collection
        if self.schema_proxy and self.schema_proxy.schemaview:
            fields = []
            for s in self.schema_proxy.schemaview.all_slots(attributes=True).values():
                if s.identifier:
                    fields.append(s.name)
            if fields:
                if len(fields) > 1:
                    raise ValueError(f"Multiple identifier fields: {fields}")
                return fields[0]
        return "id"

    def label_field(self, collection: str = None) -> str:
        field_names = self.field_names(collection)
        preferred = ["label", "name", "title"]
        for candidate in preferred:
            if candidate in field_names:
                return candidate
        return field_names[0] if field_names else "label"

    # TODO: Might be abstract as currently gives not same values for DuckDBVSSAdapter and ChromaDBAdapter
    def field_names(self, collection: str = None) -> List[str]:
        """
        Return the names of all top level fields in the database for a collection.

        :param collection:
        :return:
        """
        # TODO: use schema proxy if set
        if not self._field_names_by_collection:
            self._field_names_by_collection = {}
        collection = self._get_collection(collection)
        if collection not in self._field_names_by_collection:
            logger.debug(f"Getting field names for {collection}")
            objs = self.peek(collection=collection, limit=1)
            if not objs:
                raise ValueError(f"Collection {collection} is empty")
            fields = []
            for obj in objs:
                for f in obj:
                    if f not in fields:
                        fields.append(f)
            self._field_names_by_collection[collection] = fields
        else:
            logger.debug(f"Using cached field names for {collection}")
        return self._field_names_by_collection[collection]

    # Loading and dumping
    def dump(
        self,
        collection: str = None,
        to_file: FILE_LIKE = None,
        metadata_to_file: FILE_LIKE = None,
        format=None,
        include=None,
        **kwargs,
    ):
        """
        Dump the database to a file.

        :param collection:
        :param kwargs:
        :return:
        """
        collection_name = collection
        collection = self._get_collection(collection)
        logger.info(f"Dumping collection {self.collection_metadata(collection) }")
        metadata = self.collection_metadata(collection).model_dump(exclude_none=True)
        if format is None:
            format = "json"
        if not include:
            include = [EMBEDDINGS, DOCUMENTS, METADATAS]
            # include = ["embeddings", "documents", "metadatas"]
        if not isinstance(include, list):
            include = list(include)
        objects = list(self.find(collection=collection, include=include, **kwargs))
        if format.startswith("venomx"):
            import venomx as vx
            from venomx.tools.file_io import save_index

            cm = self.collection_metadata(collection_name, include_derived=False)
            label_field = self.label_field(collection_name)
            id_field = self.identifier_field(collection_name)

            def _label(obj):
                return obj.get(label_field, None)

            def _id(obj):
                original_id = obj.get("original_id", None)
                if original_id:
                    return original_id
                return obj.get(id_field, None)

            rows = [
                {"id": _id(obj), "text": meta["document"], "values": meta["_embeddings"]}
                for obj, _, meta in objects
            ]
            vx_objects = [{"id": _id(obj), "label": _label(obj)} for obj, _, _ in objects]
            logger.info(f"Num vx objects: {len(vx_objects)}")
            vx_index = vx.Index(
                embedding_model=vx.model.venomx.Model(name=cm.model),
                dataset=vx.model.venomx.Dataset(name=collection_name),
                embeddings_frame=pd.DataFrame(rows),
                objects=vx_objects,
            )
            save_index(vx_index, to_file)
            return
        to_file = _get_file(to_file, "w")
        metadata_to_file = _get_file(metadata_to_file, "w")
        streaming = True
        if format == "jsonl":
            writer = jsonlines.Writer(to_file)
            writer.write_all(objects)
            writer.close()
        elif format == "yamlblock":
            for obj in objects:
                to_file.write("---\n")
                yaml.dump(obj, to_file)
        else:
            streaming = False
            database = {"metadata": metadata, "objects": list(objects)}
            if format == "json":
                json.dump(database, to_file)
            elif format == "yaml":
                yaml.dump(database, to_file)
            else:
                raise ValueError(f"Unknown format {format}")
        if streaming:
            if not metadata_to_file:
                raise ValueError("Streaming dump requires metadata_to_file")
            if format == "jsonl":
                metadata_to_file.write(json.dumps(metadata))
            elif format == "yamlblock":
                metadata_to_file.write(yaml.dump(metadata))

    def dump_then_load(self, collection: str = None, target: "DBAdapter" = None):
        """
        Dump a collection to a file, then load it into another database.

        :param collection:
        :param target:
        :return:
        """
        raise NotImplementedError
