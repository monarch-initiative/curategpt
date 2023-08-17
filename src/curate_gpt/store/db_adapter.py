"""Abstract DB adapter."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel

from curate_gpt.store.metadata import CollectionMetadata
from curate_gpt.store.schema_proxy import SchemaProxy

OBJECT = Union[YAMLRoot, BaseModel, Dict]
QUERY = Union[str, YAMLRoot, BaseModel, Dict]
PROJECTION = Union[str, List[str]]
DEFAULT_COLLECTION = "default"
SEARCH_RESULT = Tuple[OBJECT, float, Optional[Dict]]


logger = logging.getLogger(__name__)


@dataclass
class DBAdapter(ABC):
    """
    Base class for adapters
    """

    path: str = None
    """Path to a location where the database is stored or disk or the network."""

    # pydantic_model: Optional[BaseModel] = None
    # """Pydantic model"""

    schema_proxy: Optional[SchemaProxy] = None
    """Schema manager"""

    @abstractmethod
    def insert(
        self, objs: Union[OBJECT, Iterable[OBJECT]], collection: str = DEFAULT_COLLECTION, **kwargs
    ):
        """
        Insert an object or list of objects into the store.

        :param objs:
        :param collection:
        :return:
        """

    def update(
        self, objs: Union[OBJECT, List[OBJECT]], collection: str = DEFAULT_COLLECTION, **kwargs
    ):
        """
        Update an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        raise NotImplementedError

    def upsert(
        self, objs: Union[OBJECT, List[OBJECT]], collection: str = DEFAULT_COLLECTION, **kwargs
    ):
        """
        Upsert an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        raise NotImplementedError

    def delete(self, id: str, collection: str = DEFAULT_COLLECTION, **kwargs):
        """
        Delete an object by its ID.

        :param id:
        :param collection:
        :return:
        """
        raise NotImplementedError

    def create_view(self, view_name: str, collection: str, expression: QUERY, **kwargs):
        """
        Create a view in the database.

        :param view:
        :return:
        """
        raise NotImplementedError

    def remove_collection(self, collection: str = DEFAULT_COLLECTION, exists_ok=False, **kwargs):
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

        :return:
        """

    @abstractmethod
    def collection_metadata(
        self, collection_name: Optional[str] = DEFAULT_COLLECTION, include_derived=False, **kwargs
    ) -> Optional[CollectionMetadata]:
        """
        Get the metadata for a collection.

        :param collection_name:
        :param include_derived: Include derived metadata, e.g. counts
        :return:
        """

    def set_collection_metadata(
        self, collection_name: Optional[str], metadata: CollectionMetadata, **kwargs
    ):
        """
        Set the metadata for a collection.

        :param collection_name:
        :return:
        """
        raise NotImplementedError

    def update_collection_metadata(self, collection_name: str, **kwargs) -> CollectionMetadata:
        """
        Update the metadata for a collection.

        :param collection_name:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self, text: str, where: QUERY = None, collection: str = DEFAULT_COLLECTION, **kwargs
    ) -> Iterator[SEARCH_RESULT]:
        """
        Query the database for a text string.

        :param text:
        :param collection:
        :param where:
        :param kwargs:
        :return:
        """

    def find(
        self,
        where: QUERY = None,
        projection: PROJECTION = None,
        collection: str = DEFAULT_COLLECTION,
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

    @abstractmethod
    def matches(self, obj: OBJECT, **kwargs) -> Iterator[SEARCH_RESULT]:
        """
        Query the database for matches to an object.

        :param obj:
        :param kwargs:
        :return:
        """

    @abstractmethod
    def lookup(self, id: str, collection: str = DEFAULT_COLLECTION, **kwargs) -> OBJECT:
        """
        Lookup an object by its ID.

        :param id:
        :param collection:
        :return:
        """

    @abstractmethod
    def peek(self, collection: str = DEFAULT_COLLECTION, limit=5, **kwargs) -> Iterator[OBJECT]:
        """
        Peek at first N objects in a collection.

        :param collection:
        :param limit:
        :return:
        """
        raise NotImplementedError

    def identifier_field(self, collection: str = None) -> str:
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

    def field_names(self, collection: str = None) -> List[str]:
        """
        Return the names of all top level fields in the database for a collection.

        :param collection:
        :return:
        """
        obj = self.peek(collection=collection, limit=1)
        if obj:
            return list(obj.keys())
        else:
            raise ValueError(f"Collection {collection} is empty")
