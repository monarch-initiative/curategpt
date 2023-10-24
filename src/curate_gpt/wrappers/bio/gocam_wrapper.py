import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar, Dict, Iterable, Iterator, Optional

import requests
from oaklib import BasicOntologyInterface, get_adapter

from curate_gpt.formatters.format_utils import camelify
from curate_gpt.wrappers import BaseWrapper
from curate_gpt.wrappers.literature import PubmedWrapper

BASE_URL = "https://go-public.s3.amazonaws.com/files/"
INDEX_URL = "https://go-public.s3.amazonaws.com/files/gocam-models.json"
MODEL_URL_TEMPLATE = "https://go-public.s3.amazonaws.com/files/go-cam/{model_id}.json"

ENABLED_BY = "RO:0002333"
PART_OF = "BFO:0000050"
OCCURS_IN = "BFO:0000066"

logger = logging.getLogger(__name__)


def _cls(obj: Dict) -> Optional[str]:
    if obj.get("type", None) == "complement":
        return None
    if "id" not in obj:
        raise ValueError(f"No ID for {obj}")
    id = obj["id"]
    pfx = id.split(":")[0]
    if pfx in ["UniProtKB", "MGI", "ZFIN", "SGD"]:
        toks = obj["label"].split(" ")
        return toks[0]
    return camelify(obj["label"])


@lru_cache
def _relation_id(p: str) -> str:
    ro_adapter = get_adapter("sqlite:obo:ro")
    lbl = ro_adapter.label(p)
    return camelify(lbl) if lbl else p


def _annotations(obj: Dict) -> Dict:
    return {a["key"]: a["value"] for a in obj["annotations"]}


MAIN_TYPES = [
    "MolecularFunction",
    "InformationBiomacromolecule",
    "BiologicalProcess",
    "CellularComponent",
    "Evidence",
    "ChemicalEntity",
    "AnatomicalEntity",
]


@dataclass
class GOCAMWrapper(BaseWrapper):
    """
    An view over a GO CAM source.
    """

    name: ClassVar[str] = "gocam"

    default_object_type = "Pathway"

    _label_adapter: BasicOntologyInterface = None

    pubmed_wrapper: PubmedWrapper = None

    ro_adapter: BasicOntologyInterface = None

    def __post_init__(self):
        self.pubmed_wrapper = PubmedWrapper()
        self.pubmed_wrapper.set_cache("gocam_pubmed_cache")
        self.ro_adapter = get_adapter("sqlite:obo:ro")

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        models = requests.get(INDEX_URL).json()
        for model in models:
            if "gocam" not in model:
                raise ValueError(f"Missing gocam in {model}")
            gocam = model["gocam"]
            yield self.object_by_id(gocam.replace("http://model.geneontology.org/", ""))

    def object_by_id(self, object_id: str) -> Optional[Dict]:
        obj = requests.get(MODEL_URL_TEMPLATE.format(model_id=object_id)).json()
        return self.object_from_dict(obj)

    def object_from_dict(self, obj: Dict) -> Optional[Dict]:
        id = obj["id"]
        logger.info(f"Processing model id: {id}")
        individuals = obj["individuals"]
        individual_to_type = {}
        individual_to_term = {}
        individual_to_annotations = {}
        sources = set()
        for i in individuals:
            typs = [_cls(x) for x in i.get("root-type", [])]
            typs = [x for x in typs if x]
            typ = None
            for t in typs:
                if t in MAIN_TYPES:
                    typ = t
                    break
            if not typ:
                logger.warning(f"Could not find type for {i}")
                continue
            individual_to_type[i["id"]] = typ
            terms = [_cls(x) for x in i.get("type", [])]
            terms = [x for x in terms if x]
            if len(terms) > 1:
                logger.warning(f"Multiple terms for {i}: {terms}")
            if not terms:
                logger.warning(f"No terms for {i}")
                continue
            individual_to_term[i["id"]] = terms[0]
            anns = _annotations(i)
            individual_to_annotations[i["id"]] = anns
            if "source" in anns:
                sources.add(anns["source"])
        activities = []
        activities_by_mf_id = defaultdict(list)
        facts_by_property = defaultdict(list)
        for fact in obj["facts"]:
            facts_by_property[fact["property"]].append(fact)
        for fact in facts_by_property.get(ENABLED_BY, []):
            s, o = fact["subject"], fact["object"]
            anns = _annotations(fact)
            evidence = anns.get("evidence", None)
            if evidence:
                pmid = individual_to_annotations.get(evidence, {}).get("source", None)
            else:
                pmid = None
            if s not in individual_to_term:
                logger.warning(f"Missing subject {s} in {individual_to_term}")
                continue
            if o not in individual_to_term:
                logger.warning(f"Missing object {o} in {individual_to_term}")
                continue
            activity = {
                "gene": individual_to_term[o],
                "activity": individual_to_term[s],
                "reference": pmid,
            }
            activities.append(activity)
            activities_by_mf_id[s].append(activity)

        for fact in facts_by_property.get(PART_OF, []):
            s, o = fact["subject"], fact["object"]
            if o not in individual_to_term:
                logger.warning(f"Missing {o} in {individual_to_term}")
                continue
            for a in activities_by_mf_id.get(s, []):
                a["process"] = individual_to_term[o]

        for fact in facts_by_property.get(OCCURS_IN, []):
            s, o = fact["subject"], fact["object"]
            if o not in individual_to_term:
                logger.warning(f"Missing {o} in {individual_to_term}")
                continue
            for a in activities_by_mf_id.get(s, []):
                a["location"] = individual_to_term[o]

        for p, facts in facts_by_property.items():
            for fact in facts:
                s, o = fact["subject"], fact["object"]
                sas = activities_by_mf_id.get(s, [])
                oas = activities_by_mf_id.get(o, [])
                if not sas or not oas:
                    continue
                if individual_to_type.get(s, None) != "MolecularFunction":
                    continue
                if individual_to_type.get(o, None) != "MolecularFunction":
                    continue
                sa = sas[0]
                oa = oas[0]
                if "relationships" not in sa:
                    sa["relationships"] = []
                rel = {
                    "type": _relation_id(p),
                    "target_gene": oa["gene"],
                    "target_activity": oa["activity"],
                }
                if rel not in sa["relationships"]:
                    sa["relationships"].append(rel)
        pmids = {a["reference"] for a in activities if "reference" in a}
        pmids = [p for p in pmids if p and p.startswith("PMID")]
        pubs = self.pubmed_wrapper.objects_by_ids(pmids)
        pubs_by_id = {p["id"]: p for p in pubs}
        for a in activities:
            if "reference" in a:
                ref = a["reference"]
                if ref in pubs_by_id:
                    a["reference_title"] = pubs_by_id[ref]["title"]
        annotations = _annotations(obj)
        return {
            "id": id,
            "title": annotations["title"],
            "species": annotations.get("in_taxon", None),
            "activities": activities,
            "publications": pubs,
        }

    @property
    def label_adapter(self) -> BasicOntologyInterface:
        """Get the label adapter."""
        if self._label_adapter is None:
            self._label_adapter = get_adapter("sqlite:obo:envo")
        return self._label_adapter

    def _labelify(self, term: Dict):
        if "label" not in term or not term["label"]:
            term["label"] = self.label_adapter.label(term["id"])
