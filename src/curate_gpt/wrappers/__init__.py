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
    "AllianceGeneWrapper",
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
    from curate_gpt.wrappers.clinical.clinvar_wrapper import ClinVarWrapper  # noqa
    from curate_gpt.wrappers.clinical.hpoa_by_pub_wrapper import HPOAByPubWrapper  # noqa
    from curate_gpt.wrappers.clinical.hpoa_wrapper import HPOAWrapper  # noqa
    from curate_gpt.wrappers.clinical.maxoa_wrapper import MAXOAWrapper  # noqa
    from curate_gpt.wrappers.general.filesystem_wrapper import FilesystemWrapper  # noqa
    from curate_gpt.wrappers.general.github_wrapper import GitHubWrapper  # noqa
    from curate_gpt.wrappers.general.google_drive_wrapper import GoogleDriveWrapper  # noqa
    from curate_gpt.wrappers.general.gspread_wrapper import GSpreadWrapper  # noqa
    from curate_gpt.wrappers.general.json_wrapper import JSONWrapper  # noqa
    from curate_gpt.wrappers.general.linkml_schema_wrapper import LinkMLSchemarapper  # noqa
    from curate_gpt.wrappers.investigation.ncbi_bioproject_wrapper import (  # noqa
        NCBIBioprojectWrapper,
    )
    from curate_gpt.wrappers.investigation.ncbi_biosample_wrapper import (  # noqa
        NCBIBiosampleWrapper,
    )
    from curate_gpt.wrappers.investigation.nmdc_wrapper import NMDCWrapper  # noqa
    from curate_gpt.wrappers.literature.bioc_wrapper import BiocWrapper  # noqa
    from curate_gpt.wrappers.literature.pmc_wrapper import PMCWrapper  # noqa
    from curate_gpt.wrappers.literature.pubmed_wrapper import PubmedWrapper  # noqa
    from curate_gpt.wrappers.literature.wikipedia_wrapper import WikipediaWrapper  # noqa
    from curate_gpt.wrappers.ontology.bioportal_wrapper import BioportalWrapper  # noqa
    from curate_gpt.wrappers.ontology.ontology_wrapper import OntologyWrapper  # noqa
    from curate_gpt.wrappers.bio.gocam_wrapper import GOCAMWrapper  # noqa
    from curate_gpt.wrappers.bio.alliance_gene_wrapper import AllianceGeneWrapper  # noqa
    from curate_gpt.wrappers.bio.mediadive_wrapper import MediaDiveWrapper  # noqa
    from curate_gpt.wrappers.bio.bacdive_wrapper import BacDiveWrapper  # noqa
    from curate_gpt.wrappers.bio.reactome_wrapper import ReactomeWrapper  # noqa
    from curate_gpt.wrappers.legal.reusabledata_wrapper import ReusableDataWrapper  # noqa

    for c in get_all_subclasses(BaseWrapper):
        if c.name == name:
            return c(**kwargs)
    raise ValueError(f"Unknown view {name}, not found in {get_all_subclasses(BaseWrapper)}")
