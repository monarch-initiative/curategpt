"""Wrapper for grouping Human Phenotype Ontology annotations by publication"""

from dataclasses import dataclass
from typing import ClassVar

from curategpt.wrappers.clinical.hpoa_wrapper import HPOAWrapper


@dataclass
class HPOAByPubWrapper(HPOAWrapper):
    """
    A wrapper over HPOA grouping by publication
    """

    name: ClassVar[str] = "hpoa_by_pub"
    default_object_type = "Publication"
    group_by_publication: bool = True
