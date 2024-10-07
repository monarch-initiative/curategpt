from typing import List

from pydantic import BaseModel, ConfigDict, Extra


class Relationship(BaseModel):
    """
    A relationship to another node.

    Corresponds to an edge in an OBO graph.
    """

    model_config = ConfigDict(protected_namespaces=())
    predicate: str
    target: str


class OntologyClass(BaseModel, extra=Extra.allow):
    """
    An ontology class.

    Corresponds to a node in an OBO graph.
    """

    model_config = ConfigDict(protected_namespaces=())
    id: str
    label: str = None
    definition: str = None
    aliases: List[str] = None
    relationships: List[Relationship] = None
    logical_definition: List[Relationship] = None
    original_id: str = None


class Ontology(BaseModel):
    """
    An ontology.

    Corresponds to an OBO graph.
    """

    model_config = ConfigDict(protected_namespaces=())
    elements: List[OntologyClass] = None
