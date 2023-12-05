"""Chat with a KB."""
import logging
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Iterable, Iterator, Optional

import requests_cache
from oaklib import BasicOntologyInterface

from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)

BASE_URL = "https://www.alliancegenome.org/api"


@dataclass
class AllianceGeneWrapper(BaseWrapper):

    """
    A wrapper over a Alliance (AGR) gene API.

    In future this may be extended to other objects.
    """

    name: ClassVar[str] = "alliance_gene"

    _label_adapter: BasicOntologyInterface = None

    default_object_type = "Gene"

    taxon_id: str = field(default="NCBITaxon:9606")

    def object_ids(self, taxon_id: str = None, **kwargs) -> Iterator[str]:
        """
        Get all gene ids for a given taxon id

        :param taxon_id:
        :param kwargs:
        :return:
        """
        session = requests_cache.CachedSession("alliance")

        if not taxon_id:
            taxon_id = self.taxon_id
        response = session.get(
            f"{BASE_URL}/geneMap/geneIDs",
            params={
                "taxonID": taxon_id,
                "rows": 50000,
            },
        )
        response.raise_for_status()
        gene_ids = response.text.split(",")
        yield from gene_ids

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        """
        All genes

        :param collection:
        :param object_ids:
        :param kwargs:
        :return:
        """
        session = requests_cache.CachedSession("alliance")

        if not object_ids:
            gene_ids = self.object_ids(**kwargs)
        else:
            gene_ids = object_ids

        for gene_id in gene_ids:
            response = session.get(f"{BASE_URL}/gene/{gene_id}")
            if response.status_code == 400:
                # unknown gene
                continue
            response.raise_for_status()
            obj = response.json()
            yield obj
