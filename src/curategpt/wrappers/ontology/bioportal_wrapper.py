from dataclasses import dataclass, field

from curate_gpt.wrappers.ontology.ontology_wrapper import OntologyWrapper
from oaklib import get_adapter


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
