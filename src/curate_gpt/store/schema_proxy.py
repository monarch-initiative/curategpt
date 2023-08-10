from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, Optional

from linkml_runtime import SchemaView
from linkml_runtime.linkml_model import SchemaDefinition
from pydantic import BaseModel


@dataclass
class SchemaProxy:
    """
    Manage connection to a schema
    """

    schema_source: Union[str, Path, SchemaDefinition] = None
    _pydantic_root_model: BaseModel = None
    _schemaview: SchemaView = None

    @property
    def pydantic_root_model(self) -> BaseModel:
        """
        Get the pydantic root model.

        If none is set, then generate it from the schema.

        :return:
        """
        if self._pydantic_root_model is None:
            from linkml.generators.pydanticgen import PydanticGenerator

            mod = PydanticGenerator(self.schema).compile_module()
            roots = [c.name for c in self.schemaview.all_classes().values() if c.tree_root]
            if not roots:
                roots = list(self.schemaview.all_classes(imports=False).keys())
            if not roots:
                raise ValueError(f"Cannot find root class in {self.schema.name}")
            if len(roots) > 1:
                raise ValueError(f"Multiple roots found in {self.schema.name}: {roots}")
            root = roots[0]
            self._pydantic_root_model = mod.__dict__[root]
        return self._pydantic_root_model

    @pydantic_root_model.setter
    def pydantic_root_model(self, value: BaseModel):
        self._pydantic_root_model = value

    @property
    def schemaview(self) -> SchemaView:
        """
        Get the schema view.

        :return:
        """
        if self._schemaview is None:
            self._schemaview = SchemaView(self.schema_source)
        return self._schemaview

    @property
    def schema(self) -> SchemaDefinition:
        """
        Get the schema

        :return:
        """
        return self.schemaview.schema

    def json_schema(self) -> Dict:
        """
        Get the JSON schema translation of the schema.
        :return:
        """
        return self.pydantic_root_model.schema()

    @property
    def name(self) -> Optional[str]:
        """
        Get the name of the schema.

        :return:
        """
        if self.schema:
            return self.schema.name
        if self.pydantic_root_model:
            return self.pydantic_root_model.__name__
