"""Abstract DB adapter."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Union, Optional

from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import SchemaDefinition
from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel

from curate_gpt.store.schema_manager import SchemaManager

OBJECT = Union[YAMLRoot, BaseModel, Dict]
QUERY = Union[str, YAMLRoot, BaseModel, Dict]
PROJECTION = Union[str, List[str]]
DEFAULT_COLLECTION = "default"
SEARCH_RESULT = Tuple[OBJECT, float, Optional[Dict]]


logger = logging.getLogger(__name__)


class CollectionMetadata(BaseModel):
    """
    Metadata about a collection
    """

    name: Optional[str] = None
    """Name of the collection"""

    description: Optional[str] = None
    """Description of the collection"""

    model: Optional[str] = None
    """Name of any ML model"""

    object_type: Optional[str] = None
    """Type of object in the collection"""


@dataclass
class DBAdapter(ABC):
    """
    Base class for adapters
    """

    path: str = None
    """Path to the database"""

    pydantic_model: Optional[BaseModel] = None
    """Pydantic model"""

    schema_manager: Optional[SchemaManager] = None
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

    def remove_collection(self, collection: str = DEFAULT_COLLECTION, **kwargs):
        """
        Remove a collection from the database.

        :param collection:
        :return:
        """
        raise NotImplementedError

    def list_collection_names(self) -> List[str]:
        """
        List all collections in the database.

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
        where: QUERY,
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


    def identifier_field(self, collection: str = None) -> str:
        if self.schema_manager and self.schema_manager.schemaview:
            fields = []
            for s in self.schema_manager.schemaview.all_slots(attributes=True).values():
                if s.identifier:
                    fields.append(s.name)
            if fields:
                if len(fields) > 1:
                    raise ValueError(f"Multiple identifier fields: {fields}")
                return fields[0]
        return "id"
