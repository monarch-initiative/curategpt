from dataclasses import dataclass
from typing import ClassVar

from curate_gpt.wrappers import HPOAWrapper


@dataclass
class HPOAByPubWrapper(HPOAWrapper):
    """
    A wrapper over HPOA grouping by publication
    """

    name: ClassVar[str] = "hpoa_by_pub"
    default_object_type = "Publication"
    group_by_publication: bool = True
