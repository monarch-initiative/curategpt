"""Simple default adapter for a object store."""

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Iterable, Iterator, List, Optional, Tuple, Union, get_origin

from pydantic import BaseModel, ConfigDict
from venomx.model.venomx import Index, Model, ModelInputMethod

from curategpt import DBAdapter
from curategpt.store.db_adapter import OBJECT, PROJECTION, QUERY, SEARCH_RESULT
from curategpt.store.metadata import Metadata

logger = logging.getLogger(__name__)


class Collection(BaseModel):

    model_config = ConfigDict(protected_namespaces=())

    objects: List[Dict] = []
    metadata: Dict = {}

    def add(self, object: Dict) -> None:
        self.objects.append(object)

    def add_metadata(self, venomx: Metadata) -> None:
        self.metadata.update(venomx)

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

    def insert(self, objs: Union[OBJECT, Iterable[OBJECT]], collection: str = None, **kwargs):
        """
        Insert an object or list of objects into the store.

        :param objs:
        :param collection:
        :return:
        """
        self._insert(objs, collection, **kwargs)

    def _insert(
        self, objs: Union[OBJECT, Iterable[OBJECT]], collection: str = None, venomx: Metadata = None
    ):
        collection_obj = self._get_collection_object(collection)
        if venomx is None:
            venomx = self.populate_venomx(
                collection=collection,
            )
        if get_origin(type(objs)) is not Dict:
            objs = [objs]

        collection_obj.add(objs)
        collection_obj.add_metadata(venomx)

    @staticmethod
    def populate_venomx(
        collection: Optional[str],
        model: Optional[str] = None,
        distance: str = None,
        object_type: str = None,
        embeddings_dimension: int = None,
        index_fields: Optional[Union[List[str], Tuple[str]]] = None,
    ) -> Metadata:
        """
        Populate venomx with data currently given when inserting

        :param collection:
        :param model:
        :param distance:
        :param object_type:
        :param embeddings_dimension:
        :param index_fields:
        :return:
        """
        venomx = Metadata(
            venomx=Index(
                id=collection,
                embedding_model=Model(name=model),
                embeddings_dimensions=embeddings_dimension,
                embedding_input_method=(
                    ModelInputMethod(fields=index_fields) if index_fields else None
                ),
            ),
            hnsw_space=distance,
            object_type=object_type,
        )
        return venomx

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
    ) -> Optional[Metadata]:
        """
        Get the metadata for a collection.

        :param collection_name:
        :param include_derived: Include derived metadata, e.g. counts
        :return:
        """
        collection_obj = self._get_collection_object(collection_name)
        md_dict = collection_obj.metadata
        cm = Metadata(**md_dict)
        if include_derived:
            cm.object_count = len(collection_obj.objects)
        return cm

    def set_collection_metadata(self, collection_name: Optional[str], metadata: Metadata, **kwargs):
        """
        Set the metadata for a collection.

        :param collection_name:
        :return:
        """
        collection_obj = self._get_collection_object(collection_name)
        # TODO: allow for now, as now embed functionality
        # if metadata.venomx.id != collection_name:
        #     raise ValueError(f"venomx.id: {metadata.venomx.id} must match collection_name {collection_name} and should not be changed")
        collection_obj.metadata = metadata.model_dump(exclude_none=True)

    def update_collection_metadata(self, collection_name: str, **kwargs) -> Metadata:
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
