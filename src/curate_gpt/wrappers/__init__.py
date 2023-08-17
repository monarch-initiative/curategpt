"""Wrappers on top of external data sources.

Wrappers allow for dynamic or static loading of an external data source
into a store.
"""

from curate_gpt.wrappers.base_wrapper import BaseWrapper
from curate_gpt.wrappers.investigation.nmdc_wrapper import NMDCWrapper
from curate_gpt.wrappers.literature.bioc_wrapper import BiocWrapper
from curate_gpt.wrappers.literature.pubmed_wrapper import PubmedWrapper
from curate_gpt.wrappers.literature.wikipedia_wrapper import WikipediaWrapper
from curate_gpt.wrappers.ontology.bioportal_wrapper import BioportalWrapper
from curate_gpt.wrappers.ontology.ontology_wrapper import OntologyWrapper
from curate_gpt.wrappers.sysbio.gocam_wrapper import GOCAMWrapper

__all__ = [
    "BaseWrapper",
    "PubmedWrapper",
    "WikipediaWrapper",
    "BiocWrapper",
    "OntologyWrapper",
    "BioportalWrapper",
    "NMDCWrapper",
    "GOCAMWrapper",
    "get_wrapper",
]


def get_wrapper(name: str, **kwargs) -> BaseWrapper:
    for c in BaseWrapper.__subclasses__():
        if c.name == name:
            return c(**kwargs)
    raise ValueError(f"Unknown view {name}, for found in {BaseWrapper.__subclasses__()}")
