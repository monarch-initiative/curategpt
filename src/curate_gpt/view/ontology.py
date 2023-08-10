from typing import List

from pydantic import BaseModel


class Relationship(BaseModel):
    """
    A relationship to another node.

    Corresponds to an edge in an OBO graph.
    """

    predicate: str
    target: str


class OntologyClass(BaseModel):
    """
    An ontology class.

    Corresponds to a node in an OBO graph.
    """

    id: str
    label: str = None
    definition: str = None
    aliases: List[str] = None
    relationships: List[Relationship] = None
    logical_definition: List[Relationship] = None


class Ontology(BaseModel):
    """
    An ontology.

    Corresponds to an OBO graph.
    """

    elements: List[OntologyClass] = None
