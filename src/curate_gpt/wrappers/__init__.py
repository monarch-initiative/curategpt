"""Wrappers on top of external data sources.

Wrappers allow for dynamic or static loading of an external data source
into a store.
"""

from curate_gpt.wrappers.base_wrapper import BaseWrapper

__all__ = [
    "BaseWrapper",
    "PubmedWrapper",
    "WikipediaWrapper",
    "BiocWrapper",
    "OntologyWrapper",
    "BioportalWrapper",
    "NMDCWrapper",
    "GOCAMWrapper",
    "NCBIBiosampleWrapper",
    "NCBIBioprojectWrapper",
    "ClinVarWrapper",
    "PMCWrapper",
    "HPOAWrapper",
    "HPOAByPubWrapper",
    "MAXOAWrapper",
    "GoogleDriveWrapper",
    "FilesystemWrapper",
    "get_wrapper",
]


def get_all_subclasses(cls):
    """Recursively get all subclasses of a given class."""
    direct_subclasses = cls.__subclasses__()
    return direct_subclasses + [
        s for subclass in direct_subclasses for s in get_all_subclasses(subclass)
    ]


def get_wrapper(name: str, **kwargs) -> BaseWrapper:
    # NOTE: ORDER DEPENDENT. TODO: fix this
    from curate_gpt.wrappers.clinical.clinvar_wrapper import ClinVarWrapper
    from curate_gpt.wrappers.clinical.hpoa_by_pub_wrapper import HPOAByPubWrapper
    from curate_gpt.wrappers.clinical.hpoa_wrapper import HPOAWrapper
    from curate_gpt.wrappers.clinical.maxoa_wrapper import MAXOAWrapper
    from curate_gpt.wrappers.general.filesystem_wrapper import FilesystemWrapper
    from curate_gpt.wrappers.general.github_wrapper import GitHubWrapper
    from curate_gpt.wrappers.general.google_drive_wrapper import GoogleDriveWrapper
    from curate_gpt.wrappers.general.gspread_wrapper import GSpreadWrapper
    from curate_gpt.wrappers.general.json_wrapper import JSONWrapper
    from curate_gpt.wrappers.general.linkml_schema_wrapper import LinkMLSchemarapper
    from curate_gpt.wrappers.investigation.ncbi_bioproject_wrapper import NCBIBioprojectWrapper
    from curate_gpt.wrappers.investigation.ncbi_biosample_wrapper import NCBIBiosampleWrapper
    from curate_gpt.wrappers.investigation.nmdc_wrapper import NMDCWrapper
    from curate_gpt.wrappers.literature.bioc_wrapper import BiocWrapper
    from curate_gpt.wrappers.literature.pmc_wrapper import PMCWrapper
    from curate_gpt.wrappers.literature.pubmed_wrapper import PubmedWrapper
    from curate_gpt.wrappers.literature.wikipedia_wrapper import WikipediaWrapper
    from curate_gpt.wrappers.ontology.bioportal_wrapper import BioportalWrapper
    from curate_gpt.wrappers.ontology.ontology_wrapper import OntologyWrapper
    from curate_gpt.wrappers.sysbio.gocam_wrapper import GOCAMWrapper

    for c in get_all_subclasses(BaseWrapper):
        if c.name == name:
            return c(**kwargs)
    raise ValueError(f"Unknown view {name}, not found in {get_all_subclasses(BaseWrapper)}")
