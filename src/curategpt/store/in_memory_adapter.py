"""Simple default adapter for a object store."""

import logging
from dataclasses import dataclass, field
from typing import (ClassVar, Dict, Iterable, Iterator, List, Optional, Union,
                    get_origin)

from pydantic import BaseModel, ConfigDict

from curategpt import DBAdapter
from curategpt.store.db_adapter import OBJECT, PROJECTION, QUERY, SEARCH_RESULT
from curategpt.store.metadata import CollectionMetadata

logger = logging.getLogger(__name__)


class Collection(BaseModel):

    model_config = ConfigDict(protected_namespaces=())

    objects: List[Dict] = []
    metadata: Dict = {}

    def add(self, object: Dict) -> None:
        self.objects.append(object)

    def delete(self, key_value: str, key: str) -> None:
        self.objects = [obj for obj in self.objects if obj[key] != key_value]


class CollectionIndex(BaseModel):

    model_config = ConfigDict(protected_namespaces=())

    collections: Dict[str, Collection] = {}

    def get_collection(self, name: str) -> Collection:
        if name not in self.collections:
            self.collections[name] = Collection()
        return self.collections[name]


@dataclass
class InMemoryAdapter(DBAdapter):
    """
    Simple in-memory adapter for a object store.
    """

    name: ClassVar[str] = "in_memory"

    collection_index: CollectionIndex = field(default_factory=CollectionIndex)

    # CUD operations

    def _get_collection_object(self, collection_name: str) -> Collection:
        """
        Get the collection object for a collection name.

        :param collection_name:
        :return:
        """
        collection_obj = self.collection_index.get_collection(self._get_collection(collection_name))
        return collection_obj

    def insert(self, objs: Union[OBJECT, Iterable[OBJECT]], collection: str = None, **kwargs):
        """
        Insert an object or list of objects into the store.

        :param objs:
        :param collection:
        :return:
        """
        collection_obj = self._get_collection_object(collection)
        if get_origin(type(objs)) is not Dict:
            objs = [objs]
        collection_obj.add(objs)

    def update(self, objs: Union[OBJECT, List[OBJECT]], collection: str = None, **kwargs):
        """
        Update an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        collection_obj = self._get_collection_object(collection)
        if isinstance(objs, OBJECT):
            objs = [objs]
        collection_obj.add(objs)

    def upsert(self, objs: Union[OBJECT, List[OBJECT]], collection: str = None, **kwargs):
        """
        Upsert an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        collection_obj = self._get_collection_object(collection)
        if isinstance(objs, OBJECT):
            objs = [objs]
        collection_obj.add(objs)

    def delete(self, id: str, collection: str = None, **kwargs):
        """
        Delete an object by its ID.

        :param id:
        :param collection:
        :return:
        """
        collection_obj = self._get_collection_object(collection)
        collection_obj.delete(id, self.identifier_field())

    # Collection operations

    def remove_collection(self, collection: str = None, exists_ok=False, **kwargs):
        """
        Remove a collection from the database.

        :param collection:
        :return:
        """
        if not exists_ok and collection not in self.collection_index.collections:
            raise ValueError(f"Collection {collection} does not exist.")
        self.collection_index.collections.pop(self._get_collection(collection), None)

    def list_collection_names(self) -> List[str]:
        """
        List all collections in the database.

        :return:
        """
        return list(self.collection_index.collections.keys())

    def collection_metadata(
        self, collection_name: Optional[str] = None, include_derived=False, **kwargs
    ) -> Optional[CollectionMetadata]:
        """
        Get the metadata for a collection.

        :param collection_name:
        :param include_derived: Include derived metadata, e.g. counts
        :return:
        """
        collection_obj = self._get_collection_object(collection_name)
        md_dict = collection_obj.metadata
        cm = CollectionMetadata(**md_dict)
        if include_derived:
            cm.object_count = len(collection_obj.objects)
        return cm

    def set_collection_metadata(
        self, collection_name: Optional[str], metadata: CollectionMetadata, **kwargs
    ):
        """
        Set the metadata for a collection.

        :param collection_name:
        :return:
        """
        collection_obj = self._get_collection_object(collection_name)
        collection_obj.metadata = metadata.dict()

    def update_collection_metadata(self, collection_name: str, **kwargs) -> CollectionMetadata:
        """
        Update the metadata for a collection.

        :param collection_name:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    # Query operations

    def search(
        self, text: str, where: QUERY = None, collection: str = None, **kwargs
    ) -> Iterator[SEARCH_RESULT]:
        """
        Query the database for a text string.

        :param text:
        :param collection:
        :param where:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def find(
        self,
        where: QUERY = None,
        projection: PROJECTION = None,
        collection: str = None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        """
        Query the database.

        :param text:
        :param collection:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def matches(self, obj: OBJECT, **kwargs) -> Iterator[SEARCH_RESULT]:
        """
        Query the database for matches to an object.

        :param obj:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def lookup(self, id: str, collection: str = None, **kwargs) -> OBJECT:
        """
        Lookup an object by its ID.

        :param id:
        :param collection:
        :return:
        """
        collection_obj = self._get_collection_object(collection)
        return collection_obj.get(id, self.identifier_field())

    def peek(self, collection: str = None, limit=5, **kwargs) -> Iterator[OBJECT]:
        """
        Peek at first N objects in a collection.

        :param collection:
        :param limit:
        :return:
        """
        collection_obj = self._get_collection_object(collection)
        yield from collection_obj.objects[:limit]

    def fetch_all_objects_memory_safe(
        self, collection: str = None, batch_size: int = 100, **kwargs
    ) -> Iterator[OBJECT]:
        pass
