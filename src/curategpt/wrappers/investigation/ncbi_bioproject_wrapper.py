"""EUtils-based wrapper for studies in NCBI."""

import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, List

import yaml

from curate_gpt.wrappers.literature.eutils_wrapper import EUtilsWrapper

logger = logging.getLogger(__name__)


@dataclass
class NCBIBioprojectWrapper(EUtilsWrapper):
    """
    A wrapper to provide a search facade over NCBI bioproject.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
    """

    name: ClassVar[str] = "ncbi_bioproject"

    eutils_db: ClassVar[str] = "bioproject"

    id_prefix: ClassVar[str] = "bioproject"

    default_object_type = "Study"

    def objects_from_dict(self, results: Dict) -> List[Dict]:
        objs = []
        for r in results["RecordSet"]["DocumentSummary"]:
            print(yaml.dump(r))
            p = r["Project"]
            d = p["ProjectDescr"]
            obj = {}

            obj["id"] = "bioproject:" + p["ProjectID"]["ArchiveID"]["@accession"]
            for k in ["Name", "Title"]:
                if k in d:
                    obj["title"] = d[k]
                    break
            obj["description"] = d["Description"]
            pt = p["ProjectType"]
            if "ProjectTypeSubmission" in pt:
                pts = pt["ProjectTypeSubmission"]
                if "Target" in pts:
                    if "Organism" in pts["Target"]:
                        obj["organism"] = pts["Target"]["Organism"]["OrganismName"]
            if "Publication" in d:
                pubs = d["Publication"]
                if isinstance(pubs, list):
                    obj["publications"] = [self._parse_publication(p) for p in pubs]
                else:
                    obj["publications"] = [self._parse_publication(pubs)]
            objs.append(obj)

        return objs

    def _parse_publication(self, pub: Dict) -> Dict:
        """Parse a publication from a bioproject."""
        if "StructuredCitation" in pub:
            sc = pub["StructuredCitation"]
            title = sc["Title"]
        else:
            title = None
        return {
            "title": title,
            "id": "pmid:" + pub["@id"],
        }
