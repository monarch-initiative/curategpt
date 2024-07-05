"""Chat with a KB."""
import logging
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Iterable, Iterator, Optional, List

import requests
import requests_cache
from oaklib import BasicOntologyInterface

from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)

BASE_URL = "https://rest.uniprot.org/uniprotkb"


@dataclass
class UniprotWrapper(BaseWrapper):

    """
    A wrapper over the UniProt API.
    """

    name: ClassVar[str] = "uniprot"

    _label_adapter: BasicOntologyInterface = None

    default_object_type = "Protein"

    #taxon_id: str = field(default="NCBITaxon:9606")
    taxon_id: Optional[str] = None

    session: requests_cache.CachedSession = field(
        default_factory=lambda: requests_cache.CachedSession("uniprot")
    )

    def objects(
            self, collection: str = None, object_ids: Iterable[str] = None, **kwargs
    ) -> Iterator[Dict]:
        """
        All proteins
        """
        object_ids = object_ids or self.object_ids()
        return self.objects_by_ids(object_ids)

    def object_ids(self, taxon_id: str = None, **kwargs) -> Iterator[str]:
        """
        Get all gene ids for a given taxon id

        :param taxon_id:
        :param kwargs:
        :return:
        """
        url = f"{BASE_URL}/search"
        session = self.session
        taxon_id = taxon_id or self.taxon_id
        if not taxon_id:
            raise ValueError("Taxon ID is required")
        taxon_id = str(taxon_id).replace("NCBITaxon:", "")
        params = {
            f"query": f"organism_id:{taxon_id} AND reviewed:true",
            # Query for E. coli using NCBI taxon ID and reviewed (Swiss-Prot) proteins
            "format": "json",  # Response format
            "size": 500,  # Number of results per page (max 500)
            "fields": "accession,id"  # Fields to retrieve
        }

        # Send the request
        logger.info(f"Getting proteins for taxon {taxon_id}")
        response = session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        entries = data.get("results", [])
        for entry in entries:
            logger.debug(f"Got entry: {entry}")
            yield entry["primaryAccession"]

    def objects_by_ids(self, object_ids: List[str]) -> List[Dict]:
        session = self.session
        objs = []
        for object_id in object_ids:
            if ":" in object_id:
                pfx, object_id = object_id.split(":")[1]
                if pfx.lower() not in ["uniprot", "uniprotkb"]:
                    raise ValueError(f"Invalid object id prefix: {pfx}")
            url = f"{BASE_URL}/{object_id}.json"
            logger.info(f"Getting protein data for {object_id} from {url}")
            response = session.get(url)
            response.raise_for_status()
            data = response.json()
            objs.append(data)
        return objs






