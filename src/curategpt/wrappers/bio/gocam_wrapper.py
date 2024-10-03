import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import ClassVar, Dict, Iterable, Iterator, Optional

import requests
import requests_cache
import yaml
from curategpt.formatters.format_utils import camelify
from curategpt.wrappers import BaseWrapper
from curategpt.wrappers.literature import PubmedWrapper
from oaklib import BasicOntologyInterface, get_adapter
from oaklib.interfaces.association_provider_interface import \
    AssociationProviderInterface

BASE_URL = "https://go-public.s3.amazonaws.com/files/"
INDEX_URL = "https://go-public.s3.amazonaws.com/files/gocam-models.json"
# MODEL_URL_TEMPLATE = "https://go-public.s3.amazonaws.com/files/go-cam/{model_id}.json"
GOCAM_ENDPOINT = "https://api.geneontology.org/api/go-cam/"

ENABLED_BY = "RO:0002333"
PART_OF = "BFO:0000050"
OCCURS_IN = "BFO:0000066"

logger = logging.getLogger(__name__)


@lru_cache
def _relation_id(p: str) -> str:
    ro_adapter = get_adapter("sqlite:obo:ro")
    lbl = ro_adapter.label(p)
    return camelify(lbl) if lbl else p


def _annotations(obj: Dict) -> Dict:
    def _normalize_property(prop: str) -> str:
        if "/" in prop:
            return prop.split("/")[-1]
        return prop

    return {_normalize_property(a["key"]): a["value"] for a in obj["annotations"]}


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

    session: requests.Session = field(default_factory=lambda: requests.Session())

    include_standard_annotations: bool = False

    def __post_init__(self):
        self.pubmed_wrapper = PubmedWrapper()
        self.pubmed_wrapper.set_cache("gocam_pubmed_cache")
        self.ro_adapter = get_adapter("sqlite:obo:ro")
        self.session = requests_cache.CachedSession("gocam_s3_cache")

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
        session = self.session
        if not object_id:
            raise ValueError(f"Missing object ID: {object_id}")
        logger.info(f"Fetch object: {object_id}")
        # response = session.get(MODEL_URL_TEMPLATE.format(model_id=object_id))
        local_id = object_id.replace("gocam:", "")
        response = session.get(f"{GOCAM_ENDPOINT}/{local_id}")
        response.raise_for_status()
        obj = response.json()
        return self.object_from_dict(obj)

    def object_from_dict(self, obj: Dict) -> Optional[Dict]:
        id = obj["id"]
        logger.info(f"Processing model id: {id}")
        logger.debug(f"Model: {yaml.dump(obj)}")
        individuals = obj["individuals"]
        individual_to_type = {}
        individual_to_term = {}
        individual_to_annotations = {}
        sources = set()

        lbl2id = {}

        derived_standard_annotations = []

        def _cls(obj: Dict) -> Optional[str]:
            if obj.get("type", None) == "complement":
                # class expression representing NOT
                return None
            if "id" not in obj:
                raise ValueError(f"No ID for {obj}")
            id = obj["id"]
            pfx = id.split(":")[0]
            if pfx in ["UniProtKB", "MGI", "ZFIN", "SGD"]:
                toks = obj["label"].split(" ")
                n = toks[0]
            else:
                n = camelify(obj["label"])
            lbl2id[n] = str(id)
            return n

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
        relationships = []
        gene_ids = set()
        process_ids = set()
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
            gene_id = individual_to_term[o]
            # we only include a single pmid for each activity
            activity = {
                "gene": gene_id,
                "activity": individual_to_term[s],
                "reference": pmid,
            }
            gene_ids.add(gene_id)
            activities.append(activity)
            activities_by_mf_id[s].append(activity)

        for fact in facts_by_property.get(PART_OF, []):
            s, o = fact["subject"], fact["object"]
            if o not in individual_to_term:
                logger.warning(f"Missing {o} in {individual_to_term}")
                continue
            for a in activities_by_mf_id.get(s, []):
                a["process"] = individual_to_term[o]
                process_ids.add(individual_to_term[o])

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
                relationships.append(rel)
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
        assocs = None
        if self.include_standard_annotations:
            std_to_activity = {}
            for a in activities:
                gene_id = a["gene"]
                activity_id = a.get("activity", None)
                if activity_id:
                    derived_standard_annotations.append((gene_id, activity_id))
                    std_to_activity[(gene_id, activity_id)] = a
                location_id = a.get("location", None)
                if location_id:
                    derived_standard_annotations.append((gene_id, location_id))
                    std_to_activity[(gene_id, location_id)] = a
                process_id = a.get("process", None)
                if process_id:
                    derived_standard_annotations.append((gene_id, process_id))
                    std_to_activity[(gene_id, process_id)] = a
            actual_gene_ids = []
            for k, v in lbl2id.items():
                if k in gene_ids:
                    actual_gene_ids.append(v)
            assocs = []
            adapter = get_adapter("amigo:")
            if not isinstance(adapter, AssociationProviderInterface):
                raise ValueError(f"Invalid adapter: {adapter}")
            # std_anns_accounted_for = set()
            for assoc in adapter.associations(actual_gene_ids):
                gene_obj = {
                    "id": assoc.subject,
                    "label": assoc.subject_label,
                }
                term_obj = {
                    "id": assoc.object,
                    "label": assoc.object_label,
                }
                gene_ref = _cls(gene_obj)
                term_ref = _cls(term_obj)
                ann_pubs = [str(x) for x in assoc.publications]
                ann_pubs = [p for p in ann_pubs if p.startswith("PMID") or p == "GO_REF:0000033"]
                assoc_obj = {
                    "gene": gene_ref,
                    "term": term_ref,
                    # "evidence": assoc.evidence_type,
                    "is_pub_in_gocam": len(set(ann_pubs).intersection(pmids)) > 0,
                }
                if ann_pubs:
                    assoc_obj["publication"] = ann_pubs[0]
                if (gene_ref, term_ref) in derived_standard_annotations:
                    assoc_obj["is_annotation_in_gocam"] = True
                    std_to_activity[(gene_ref, term_ref)]["has_standard_annotation"] = True
                assocs.append(assoc_obj)
            for a in activities:
                if "has_standard_annotation" not in a:
                    a["has_standard_annotation"] = False
        lbl2id = {k: v for k, v in lbl2id.items() if v.split(":")[0] not in ["ECO", "CARO", "obo"]}
        cam = {
            "id": id,
            "title": annotations["title"],
            "species": annotations.get("in_taxon", None),
            "provided_by": annotations.get("providedBy", None),
            "activities": activities,
            "publications": pubs,
            "idmap": lbl2id,
            "stats": {
                "num_genes": len(gene_ids),
                "num_processes": len(process_ids),
                "num_activities": len(activities),
                "num_relationships": len(relationships),
                "num_publications": len(pmids),
            },
        }
        if assocs:
            cam["associations"] = assocs
        return cam

    @property
    def label_adapter(self) -> BasicOntologyInterface:
        """Get the label adapter."""
        if self._label_adapter is None:
            self._label_adapter = get_adapter("sqlite:obo:envo")
        return self._label_adapter

    def _labelify(self, term: Dict):
        if "label" not in term or not term["label"]:
            term["label"] = self.label_adapter.label(term["id"])
