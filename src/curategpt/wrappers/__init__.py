"""
Wrappers on top of external data sources.

Wrappers allow for dynamic or static loading of an external data source
into a store.
"""

from curategpt.wrappers.base_wrapper import BaseWrapper

__all__ = [
    "BaseWrapper",
    "PubmedWrapper",
    "WikipediaWrapper",
    "BiocWrapper",
    "OntologyWrapper",
    "BioportalWrapper",
    "NMDCWrapper",
    "GOCAMWrapper",
    "JGIWrapper",
    "AllianceGeneWrapper",
    "NCBIBiosampleWrapper",
    "NCBIBioprojectWrapper",
    "OmicsDIWrapper",
    "UniprotWrapper",
    "OBOFormatWrapper",
    "ClinVarWrapper",
    "PMCWrapper",
    "HPOAWrapper",
    "HPOAByPubWrapper",
    "MAXOAWrapper",
    "GoogleDriveWrapper",
    "FAIRSharingWrapper",
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
    from curategpt.wrappers.bio.alliance_gene_wrapper import \
        AllianceGeneWrapper  # noqa
    from curategpt.wrappers.bio.bacdive_wrapper import BacDiveWrapper  # noqa
    from curategpt.wrappers.bio.gocam_wrapper import GOCAMWrapper  # noqa
    from curategpt.wrappers.bio.mediadive_wrapper import \
        MediaDiveWrapper  # noqa
    from curategpt.wrappers.bio.omicsdi_wrapper import OmicsDIWrapper  # noqa
    from curategpt.wrappers.bio.reactome_wrapper import \
        ReactomeWrapper  # noqa
    from curategpt.wrappers.bio.uniprot_wrapper import UniprotWrapper  # noqa
    from curategpt.wrappers.clinical.clinvar_wrapper import \
        ClinVarWrapper  # noqa
    from curategpt.wrappers.clinical.ctgov_wrapper import \
        ClinicalTrialsWrapper  # noqa
    from curategpt.wrappers.clinical.hpoa_by_pub_wrapper import \
        HPOAByPubWrapper  # noqa
    from curategpt.wrappers.clinical.hpoa_wrapper import HPOAWrapper  # noqa
    from curategpt.wrappers.clinical.maxoa_wrapper import MAXOAWrapper  # noqa
    from curategpt.wrappers.general.filesystem_wrapper import \
        FilesystemWrapper  # noqa
    from curategpt.wrappers.general.github_wrapper import \
        GitHubWrapper  # noqa
    from curategpt.wrappers.general.google_drive_wrapper import \
        GoogleDriveWrapper  # noqa
    from curategpt.wrappers.general.gspread_wrapper import \
        GSpreadWrapper  # noqa
    from curategpt.wrappers.general.json_wrapper import JSONWrapper  # noqa
    from curategpt.wrappers.general.linkml_schema_wrapper import \
        LinkMLSchemarapper  # noqa
    from curategpt.wrappers.investigation.ess_deepdive_wrapper import \
        ESSDeepDiveWrapper  # noqa
    from curategpt.wrappers.investigation.fairsharing_wrapper import \
        FAIRSharingWrapper  # noqa
    from curategpt.wrappers.investigation.jgi_wrapper import \
        JGIWrapper  # noqa
    from curategpt.wrappers.investigation.ncbi_bioproject_wrapper import \
        NCBIBioprojectWrapper  # noqa
    from curategpt.wrappers.investigation.ncbi_biosample_wrapper import \
        NCBIBiosampleWrapper  # noqa
    from curategpt.wrappers.investigation.nmdc_wrapper import \
        NMDCWrapper  # noqa
    from curategpt.wrappers.legal.reusabledata_wrapper import \
        ReusableDataWrapper  # noqa
    from curategpt.wrappers.literature.bioc_wrapper import BiocWrapper  # noqa
    from curategpt.wrappers.literature.pmc_wrapper import PMCWrapper  # noqa
    from curategpt.wrappers.literature.pubmed_wrapper import \
        PubmedWrapper  # noqa
    from curategpt.wrappers.literature.wikipedia_wrapper import \
        WikipediaWrapper  # noqa
    from curategpt.wrappers.ontology.bioportal_wrapper import \
        BioportalWrapper  # noqa
    from curategpt.wrappers.ontology.oboformat_wrapper import \
        OBOFormatWrapper  # noqa
    from curategpt.wrappers.ontology.ontology_wrapper import \
        OntologyWrapper  # noqa

    for c in get_all_subclasses(BaseWrapper):
        if c.name == name:
            return c(**kwargs)
    raise ValueError(f"Unknown view {name}, not found in {get_all_subclasses(BaseWrapper)}")
