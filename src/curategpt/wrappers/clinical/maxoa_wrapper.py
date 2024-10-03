"""Chat with a KB."""

import logging
from copy import deepcopy
from csv import DictReader
from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar, Dict, Iterable, Iterator, Optional, TextIO

import requests
from oaklib import BasicOntologyInterface, get_adapter

from curate_gpt.wrappers import BaseWrapper
from curate_gpt.wrappers.literature.pubmed_wrapper import PubmedWrapper

logger = logging.getLogger(__name__)

MAP = {
    "disease_id": "disease",
    "disease_name": "disease_label",
    "hpo_id": "phenotype",
    "maxo_id": "treatment",
    "maxo_label": "treatment_label",
}


def filter_header(row) -> bool:
    return row[0] != "#"


def stream_filtered_lines(response):
    """Generator to yield non-comment lines from a streaming response."""
    buffer = ""
    for chunk in response.iter_content(chunk_size=8192):
        # Decode the chunk of bytes to string
        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if not line.strip().startswith("#"):
                yield line


@lru_cache
def hpo_ont_adapter():
    return get_adapter("sqlite:obo:hp")


@lru_cache
def maxo_ont_adapter():
    return get_adapter("sqlite:obo:maxp")


@lru_cache
def term_label(identifier: str) -> str:
    return hpo_ont_adapter().label(identifier)


@dataclass
class MAXOAWrapper(BaseWrapper):
    """
    A wrapper over HPOA

    """

    name: ClassVar[str] = "maxoa"

    source_url: ClassVar[str] = (
        "https://raw.githubusercontent.com/monarch-initiative/"
        "maxo-annotations/master/annotations/maxo-annotations.tsv"
    )

    expand_publications: bool = True

    _label_adapter: BasicOntologyInterface = None

    default_object_type = "DiseaseTreatmentAssociation"

    pubmed_wrapper: PubmedWrapper = None

    group_by_publication: bool = True

    def __post_init__(self):
        from curate_gpt.wrappers.literature.pubmed_wrapper import PubmedWrapper

        self.pubmed_wrapper = PubmedWrapper()
        # reuse hpoa cache
        self.pubmed_wrapper.set_cache("hpoa_pubmed_cache")

    def objects(
        self,
        collection: str = None,
        object_ids: Optional[Iterable[str]] = None,
        source_locator=None,
        **kwargs,
    ) -> Iterator[Dict]:
        # open a file handle from a URL using requests
        if source_locator is None:
            source_locator = self.source_locator
        if source_locator is None:
            source_locator = self.source_url
        if source_locator.startswith("http"):
            logger.info(f"Fetching {source_locator}")
            with requests.get(source_locator, stream=True) as response:
                response.raise_for_status()  # Raise an error for failed requests
                reader = DictReader(stream_filtered_lines(response), delimiter="\t")
                yield from self.objects_from_rows(reader)
        else:
            with open(source_locator) as file:
                reader = DictReader(filter(filter_header, file), delimiter="\t")
                yield from self.objects_from_rows(reader)

    def objects_from_rows(self, rows: Iterable[Dict]) -> Iterator[Dict]:
        by_pub = {}
        for row in rows:
            row = {MAP.get(k, k): v for k, v in row.items()}
            row["phenotype_label"] = term_label(row["phenotype"])
            refs = [ref for ref in row["citation"].split(";") if ref != "PMID:UNKNOWN"]
            if self.expand_publications:
                pmids = [ref for ref in refs if ref.startswith("PMID")]
                logger.debug(f"Expanding {refs}, pmids={pmids}")
                if pmids:
                    pubs = self.pubmed_wrapper.objects_by_ids(pmids)
                    row["publications"] = pubs
                    for pub in row["publications"]:
                        pub_id = pub["id"]
                        if pub_id not in by_pub:
                            by_pub[pub_id] = deepcopy(pub)
                            by_pub[pub_id]["associations"] = []
                            logger.info(f"Adding {pub_id} to by_pub, {len(by_pub)} total")
                        by_pub[pub_id]["associations"].append(
                            {k: v for k, v in row.items() if k != "publications"}
                        )
            if not self.group_by_publication:
                yield row
        if self.group_by_publication:
            for pub in by_pub.values():
                yield pub

    def objects_from_file(self, file: TextIO) -> Iterator[Dict]:
        rows = DictReader(filter(filter_header, file), delimiter="\t")
        yield from self.objects_from_rows(rows)
