"""Chat with a KB."""

import logging
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

import requests
import requests_cache
from bs4 import BeautifulSoup
from defusedxml.ElementTree import fromstring
from eutils import Client

from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

RATE_LIMIT_DELAY = 1.0


def extract_all_text(element):
    text = element.text or ""
    for subelement in element:
        text += extract_all_text(subelement)
        text += subelement.tail if subelement.tail else ""
    return text


def extract_text_from_xml(xml_content):
    root = fromstring(xml_content)
    return extract_all_text(root).strip()


# TODO: rewrite to subclass EUtilsWrapper
@dataclass
class PubmedWrapper(BaseWrapper):
    """
    A wrapper to provide a search facade over PubMed.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.

    This can be used to restrict search to a journal; first find the journal:
    https://www.ncbi.nlm.nih.gov/nlmcatalog/?term=%22Case+Reports+in+Genetics%22

    E.g.

    `NLM Title Abbreviation: Case Rep Genet`

    Add this to the where clause:

    {"ta": "Case Rep Genet"}

    On the command line:

    .. code-block:: bash

        curategpt view search -V pubmed -I "where: {ta: Case Rep Genet}" "leukodystrophy and ischemia"
    """

    name: ClassVar[str] = "pubmed"

    eutils_client: Client = None

    session: requests.Session = field(default_factory=lambda: requests.Session())

    where: Optional[Dict] = None

    email: Optional[str] = None

    ncbi_key: Optional[str] = None

    is_fetch_full_text: Optional[bool] = None

    _uses_cache: bool = None

    def set_cache(self, name: str) -> None:
        self.session = requests_cache.CachedSession(name)
        self._uses_cache = True

    def external_search(
        self, text: str, expand: bool = True, where: Optional[Dict] = None, **kwargs
    ) -> List:
        if expand:
            logger.info(f"Expanding search term: {text} to create pubmed query")
            model = self.extractor.model
            response = model.prompt(
                text,
                system="""
                Take the specified search text, and expand it to a list
                of terms used to construct a PubMed query. You will return results as
                semi-colon separated list of the most relevant terms. Make sure to
                include all relevant concepts in the returned terms.""",
            )
            terms = response.text().split(";")
            search_term = " OR ".join(terms)
        else:
            search_term = text
        if not where and self.where:
            where = self.where
        if where:
            whereq = " AND ".join([f"{v}[{k}]" for k, v in where.items()])
            search_term = f"({search_term}) AND {whereq}"
        logger.info(f"Constructed search term: {search_term}")
        # Parameters for the request
        params = {
            "db": "pubmed",
            "term": search_term,
            "retmax": 100,
            "sort": "relevance",
            "retmode": "json",
        }

        # Note: we don't cache this call as there could be many
        # different search terms
        response = requests.get(ESEARCH_URL, params=params)
        time.sleep(RATE_LIMIT_DELAY)
        if response.status_code != 200:
            logger.error(f"Failed to search for {text}; params: {params}")
            return []
        data = response.json()

        # Extract PubMed IDs from the response
        if "esearchresult" not in data:
            logger.error(f"Failed to find results for {text}")
            logger.error(f"Data: {data}")
            return []
        pubmed_ids = data["esearchresult"]["idlist"]

        if not pubmed_ids:
            logger.warning(f"No results with {search_term}")
            if expand:
                logger.info("Trying again without expansion")
                return self.external_search(text, expand=False, **kwargs)
            else:
                logger.error(f"Failed to find results for {text}")
                return []

        logger.info(f"Found {len(pubmed_ids)} results: {pubmed_ids}")
        return self.objects_by_ids(pubmed_ids)

    def objects_by_ids(self, object_ids: List[str]) -> List[Dict]:
        pubmed_ids = sorted([x.replace("PMID:", "") for x in object_ids])
        session = self.session
        logger.debug(f"Using session: {session} [cached: {self._uses_cache} for {pubmed_ids}")

        # Parameters for the efetch request
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(pubmed_ids),  # Combine PubMed IDs into a comma-separated string
            "rettype": "medline",
            "retmode": "text",
        }
        efetch_response = session.get(EFETCH_URL, params=efetch_params)
        if not self._uses_cache or not efetch_response.from_cache:
            # throttle if not using cache or if not cached
            logger.debug(f"Sleeping for {RATE_LIMIT_DELAY} seconds")
            time.sleep(RATE_LIMIT_DELAY)
        if not efetch_response.ok:
            logger.error(f"Failed to fetch data for {pubmed_ids}")
            raise ValueError(
                f"Failed to fetch data for {pubmed_ids} using {session} and {efetch_params}"
            )
        medline_records = efetch_response.text

        # Parsing titles and abstracts from the MEDLINE records
        parsed_data = []
        current_record = {}
        current_field = None

        for line in medline_records.split("\n"):
            if line.startswith("PMID- "):
                current_field = "id"
                current_record[current_field] = "PMID:" + line.replace("PMID- ", "").strip()
            if line.startswith("PMC - "):
                current_field = "pmcid"
                pmcid = line.replace("PMC - ", "").strip()
                current_record[current_field] = "PMCID:" + pmcid
                if self.is_fetch_full_text:
                    full_text = self.pmc_full_text(pmcid)
                    # full_text = self.fetch_full_text(pmcid)
                    current_record["full_text"] = full_text
            elif line.startswith("TI  - "):
                current_field = "title"
                current_record[current_field] = line.replace("TI  - ", "").strip()
            elif line.startswith("AB  - "):
                current_field = "abstract"
                current_record[current_field] = line.replace("AB  - ", "").strip()
            elif line.startswith("    "):  # Continuation of the previous field
                if current_field and current_field in current_record:
                    current_record[current_field] += " " + line.strip()
            else:
                current_field = None

            if line == "":
                if current_record:
                    parsed_data.append(current_record)
                    current_record = {}
        return parsed_data

    def fetch_pmcid(self, pmid: str) -> Optional[str]:
        pmid = pmid.replace("PMID:", "")
        session = self.session
        params = {"db": "pmc", "linkname": "pubmed_pmc", "id": pmid}
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        response = session.get(url, params=params)
        root = fromstring(response.content)

        pmcid = None
        for link_set in root.findall(".//LinkSet"):
            for link in link_set.findall(".//Link"):
                pmcid = link.find("./Id").text
                if pmcid:
                    return f"PMC:PMC{pmcid}"
        return None

    def fetch_full_text(self, object_id: str) -> Optional[str]:
        # OLD
        session = self.session
        if object_id.startswith("PMID:"):
            pmcid = self.fetch_pmcid(object_id)
        else:
            pmcid = object_id

        if not pmcid:
            return None

        # PMC is a banana - get rid of the PMC prefix as well as local prefix
        pmcid = pmcid.replace("PMC:", "")
        pmcid = pmcid.replace("PMC", "")
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmcid}"
        response = session.get(url)
        root = fromstring(response.content)

        for record in root.findall(".//record"):
            for link in record.findall(".//link"):
                format_type = link.attrib.get("format")
                download_url = link.attrib.get("href")
                if format_type == "xml":
                    xml_response = requests.get(download_url)
                    return xml_response.text
                elif format_type == "tgz":
                    # make a named temp file
                    local_file_path = tempfile.NamedTemporaryFile().name
                    parsed_url = urlparse(download_url)
                    if parsed_url.scheme not in ["http", "https", "ftp"]:
                        continue
                    urlretrieve(download_url, local_file_path)  # noqa S310

                    # Open and extract the tar.gz file
                    with tarfile.open(local_file_path, "r:gz") as tar:
                        for member in tar.getmembers():
                            if member.name.endswith(".xml") or member.name.endswith(".nxml"):
                                f = tar.extractfile(member)
                                xml_str = f.read().decode("utf-8")
                                return extract_text_from_xml(xml_str)
                                # return xml_str
        return None

    def pmc_full_text(self, object_id: str) -> Optional[str]:
        if object_id.startswith("PMID:"):
            pmcid = self.fetch_pmcid(object_id)
        else:
            pmcid = object_id
        fulltext = self.pmc_xml(pmcid)
        logger.debug(f"FULL TEXT XML: {fulltext}")
        fullsoup = BeautifulSoup(fulltext, "xml")
        if fullsoup.find("pmc-articleset").find("article").find("body"):
            body = fullsoup.find("pmc-articleset").find("article").find("body").text
            return body
        else:
            return None

    def pmc_xml(self, pmc_id: str) -> str:
        """Get the text of one PubMed Central entry.

        Don't parse further here - just get the raw response.
        :param pmc_id: List of PubMed IDs, or string with single PMID
        :return: the text of a single entry as XML
        """
        session = self.session

        pmc_id = pmc_id.replace("PMC:", "")

        params = {
            "db": "pmc",
            "id": pmc_id,
            "rettype": "xml",
            "retmode": "xml",
        }

        if self.email and self.ncbi_key:
            params["api_key"] = self.ncbi_key
            params["email"] = self.email

        efetch_response = session.get(EFETCH_URL, params=params)
        if not self._uses_cache or not efetch_response.from_cache:
            # throttle if not using cache or if not cached
            logger.debug(f"Sleeping for {RATE_LIMIT_DELAY} seconds")
            time.sleep(RATE_LIMIT_DELAY)
        if not efetch_response.ok:
            logger.error(f"Failed to fetch data for {pmc_id}")
            raise ValueError(f"Failed to fetch data for {pmc_id} using {session} and {params}")
        return efetch_response.text
