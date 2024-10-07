from dataclasses import dataclass, field

from oaklib import get_adapter

from curategpt.wrappers.ontology.ontology_wrapper import OntologyWrapper


@dataclass
class BioportalWrapper(OntologyWrapper):
    """
    A wrapper over the Bioportal API.

    This makes use of OAK.
    """

    fetch_definitions: bool = field(default=False)
    fetch_relationships: bool = field(default=False)

    def __post_init__(self):
        if not self.oak_adapter:
            self.oak_adapter = get_adapter("bioportal:")
