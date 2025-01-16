import json
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict
from venomx.model.venomx import Index

"""
ChromaDB Constraints:
    Metadata Must Be Scalar: ChromaDB only accepts metadata values that are scalar types (str, int, float, bool).
    No None Values: Metadata fields cannot have None as a value.
DuckDB Capabilities:
    Nested Objects Supported: DuckDB can handle nested objects directly within metadata.
"""


class Metadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # Application-level field for 'duckdb' and to keep pydantic advantages in code for 'chromadb'
    venomx: Optional[Index] = None
    """
    Retains the complex venomx Index object for internal application use.
    Index is the main object of venomx
    https://github.com/cmungall/venomx
    """

    # Serialized field to store venomx in adapters that require scalar metadata (e.g., 'chromadb')
    _venomx: Optional[str] = None
    """Stores the serialized JSON string of the venomx object for ChromaDB."""

    hnsw_space: Optional[str] = None
    """Space used for hnsw index (e.g. 'cosine')"""

    object_type: Optional[str] = None
    """Type of object in the collection"""

    object_count: Optional[int] = None

    @classmethod
    def deserialize_venomx_metadata_from_adapter(cls, metadata_dict: dict, adapter: str) -> Dict:
        """
        Create a Metadata instance from adapter-specific metadata dictionary.
        ChromaDB: _venomx is deserialized back into venomx. (str to dict)
        DuckDB: venomx is accessed directly as a nested object.
        :param metadata_dict: Metadata dictionary from the adapter.
        :param adapter: Adapter name (e.g., 'chroma', 'duckdb').
        :return: Metadata instance.
        """
        if adapter == "chromadb":
            # Deserialize '_venomx' (str) back into 'venomx' (dict)
            if "_venomx" in metadata_dict:
                venomx_json = metadata_dict.pop("_venomx")
                metadata_dict["venomx"] = Index(**json.loads(venomx_json))
        # for 'duckdb', 'venomx' remains as is
        if adapter == "duckdb":
            metadata_dict = metadata_dict
        return cls(**metadata_dict)

    def serialize_venomx_metadata_for_adapter(self, adapter: str) -> dict:
        """
        Convert the Metadata instance to a dictionary suitable for the specified adapter.
        ChromaDB: venomx is serialized into _venomx before storing. (dict to str)
        DuckDB: venomx remains as an Index object without serialization.
        :param adapter: Adapter name (e.g., 'chroma', 'duckdb').
        :return: Metadata dictionary.
        """
        if adapter == "chromadb":
            # Serialize 'venomx' (dict) into '_venomx' (str)
            metadata_dict = self.model_dump(
                exclude={"venomx"}, exclude_unset=True, exclude_none=True
            )
            if self.venomx:
                metadata_dict["_venomx"] = json.dumps(self.venomx.model_dump())
            return metadata_dict
        elif adapter == "duckdb":
            metadata_dict = self.model_dump(
                exclude={"_venomx"}, exclude_unset=True, exclude_none=True
            )
            return metadata_dict
        else:
            raise ValueError(f"Unsupported adapter: {adapter}")
