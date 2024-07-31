"""Chat with a KB."""

import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, List

from curate_gpt.wrappers.literature.eutils_wrapper import EUtilsWrapper

logger = logging.getLogger(__name__)


@dataclass
class NCBIBiosampleWrapper(EUtilsWrapper):
    """
    A wrapper to provide a search facade over NCBI Biosample.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
    """

    name: ClassVar[str] = "ncbi_biosample"

    eutils_db: ClassVar[str] = "biosample"

    id_prefix: ClassVar[str] = "biosample"

    default_object_type = "Sample"

    def objects_from_dict(self, results: Dict) -> List[Dict]:
        samples = []
        for s in results["BioSampleSet"]["BioSample"]:
            d = s["Description"]
            sample = {}
            sample["id"] = f"biosample:{s['@accession']}"
            sample["title"] = d["Title"]
            sample["organism"] = d["Organism"]["@taxonomy_name"]
            sample["package"] = s["Package"]["@display_name"]
            if not s["Attributes"]:
                logger.warning(f"Skipping sample with no attributes: {s}")
                continue
            for a in s["Attributes"]["Attribute"]:
                if isinstance(a, str):
                    logger.warning(f"Skipping attribute: {a} in {s['Attributes']}")
                    continue
                a_name = a.get("@harmonized_name", a.get("@attribute_name"))
                sample[a_name] = a["#text"]
            samples.append(sample)

        return samples
