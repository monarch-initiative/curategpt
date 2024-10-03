"""Retrieval Augmented Generation (RAG) Base Class."""

import logging
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import llm
from curate_gpt.store.schema_proxy import SchemaProxy
from linkml_runtime import SchemaView
from pydantic import BaseModel as BaseModel
from pydantic import ConfigDict

logger = logging.getLogger(__name__)


class AnnotatedObject(BaseModel):
    """
    Annotated object shadows a basic dictionary object
    """

    model_config = ConfigDict(protected_namespaces=())

    # object: Union[Dict[str, Any], List[Dict[str, Any]]] = {} - TODO: pydantic bug?
    object: Any = {}
    annotations: Dict[str, Any] = {}
    key_values: Dict[str, "AnnotatedObject"] = {}

    def as_single_object(self) -> Dict[str, Any]:
        """
        Return as a standard dictionary object.

        Each annotation is prefixed with an underscore.

        :return: dictionary object
        """
        object = copy(self.object)
        for key, value in self.annotations.items():
            object[f"_{key}"] = value
        for key, value in self.key_values.items():
            object[key] = value.as_single_object()
        return object

    @property
    def text(self) -> Optional[str]:
        """
        Get the text annotation of the object.

        :return:
        """
        return self.annotations.get("text", None)


EXAMPLE = Tuple[str, AnnotatedObject]


@dataclass
class Extractor(ABC):
    # db_adapter: DBAdapter = None
    schema_proxy: SchemaProxy = None
    model_name: str = None
    _model = None
    api_key: str = None
    raise_error_if_unparsable: bool = False

    @abstractmethod
    def extract(
        self, text: str, target_class: str, examples: List[AnnotatedObject] = None, **kwargs
    ) -> AnnotatedObject:
        """
        Schema-guided extraction

        :param text:
        :param kwargs:
        :return:
        """

    @property
    def model(self):
        """
        Get the model

        :param model_name:
        :return:
        """
        if self._model is None:
            model = llm.get_model(self.model_name or "gpt-3.5-turbo")
            if model.needs_key:
                model.key = llm.get_key(self.api_key, model.needs_key, model.key_env_var)
            self._model = model
        return self._model

    @property
    def schemaview(self) -> SchemaView:
        return self.schema_proxy.schemaview

    @property
    def pydantic_root_model(self) -> BaseModel:
        return self.schema_proxy.pydantic_root_model

    def deserialize(self, text: str, **kwargs) -> AnnotatedObject:
        """
        Deserialize text into an annotated object

        :param text:
        :return:
        """
        raise NotImplementedError
