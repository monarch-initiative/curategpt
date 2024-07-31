"""EUtils-based wrapper for studies in NCBI."""

import logging
from dataclasses import dataclass
from typing import ClassVar, Dict, List

from curate_gpt.wrappers.literature.eutils_wrapper import EUtilsWrapper

logger = logging.getLogger(__name__)


@dataclass
class ClinVarWrapper(EUtilsWrapper):
    """
    A wrapper to provide a search facade over NCBI clinvar.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
    """

    name: ClassVar[str] = "clinvar"

    eutils_db: ClassVar[str] = "clinvar"

    id_prefix: ClassVar[str] = "ClinVar"

    default_object_type = "Variant"

    fetch_tool = "esummary"

    def objects_from_dict(self, results: Dict) -> List[Dict]:
        objs = []
        for r in results["eSummaryResult"]["DocumentSummarySet"]["DocumentSummary"]:
            obj = {}
            obj["id"] = "clinvar:" + r["accession"]
            obj["clinical_significance"] = r["clinical_significance"]["description"]
            obj["clinical_significance_status"] = r["clinical_significance"]["review_status"]
            obj["gene_sort"] = r["gene_sort"]
            if "genes" in r and r["genes"]:
                if "gene" in r["genes"]:
                    genes = r["genes"]["gene"]
                    if not isinstance(genes, list):
                        genes = [genes]
                    obj["genes"] = [self._gene_from_dict(g) for g in genes]
            obj["type"] = r["obj_type"]
            obj["protein_change"] = r["protein_change"]
            obj["title"] = r["title"]
            obj["traits"] = [
                self._trait_from_dict(t) for t in r["trait_set"]["trait"] if isinstance(t, dict)
            ]
            objs.append(obj)
        return objs

    def _gene_from_dict(self, gene: Dict) -> Dict:
        return {
            "id": "NCBIGene:" + gene["GeneID"],
            "symbol": gene["symbol"],
        }

    def _trait_from_dict(self, trait: Dict) -> Dict:
        obj = {
            "name": trait["trait_name"],
        }
        xrefsp = trait.get("trait_xrefs", None)
        if xrefsp:
            xrefs = xrefsp["trait_xref"]
            if not isinstance(xrefs, list):
                xrefs = [xrefs]
            obj["xrefs"] = [self._xref_from_dict(x) for x in xrefs]
        return obj

    def _xref_from_dict(self, xref: Dict) -> str:
        db_id = xref["db_id"]
        db_source = xref["db_source"]
        if db_id.lower().startswith(db_source.lower()):
            # no bananas
            return db_id
        else:
            return f"{db_source}:{db_id}"
