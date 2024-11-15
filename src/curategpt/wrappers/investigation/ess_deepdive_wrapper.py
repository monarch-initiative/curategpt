"""Wrapper for the ESS-DeepDive API."""

import logging
from dataclasses import dataclass, field
from time import sleep
from typing import ClassVar, Dict, Iterable, Iterator, List, Optional

import requests
import requests_cache

from curategpt.wrappers import BaseWrapper

BASE_URL = "https://fusion.ess-dive.lbl.gov/api/v1/deepdive"
URL = BASE_URL + "?rowStart={cursor}&pageSize={limit}"


def _get_records_chunk(session: requests_cache.CachedSession, cursor=1, limit=200) -> dict:
    """
    Get a chunk of records from ESSDeepDive.

    :param cursor:
    :return:
    """
    url = URL.format(limit=limit, cursor=cursor)
    response = session.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Could not download records from {url}")


def get_records(
    session: requests_cache.CachedSession, cursor=1, limit=200, maximum: int = None
) -> Iterator[dict]:
    """
    Iterate through all records in ESSDeepDive and download them.

    :param cursor:
    :return:
    """
    # note: do NOT do this recursively, as it will cause a stack overflow
    initial_chunk = _get_records_chunk(session, cursor=cursor, limit=limit)
    yield from initial_chunk["results"]
    while True:
        cursor += 1
        next_chunk = _get_records_chunk(session, cursor=cursor, limit=limit)
        yield from next_chunk["results"]
        pc = next_chunk["pageCount"]
        logger.warning(f"Got {pc} pages")
        if not pc:
            break
        sleep(0.1)


logger = logging.getLogger(__name__)


@dataclass
class ESSDeepDiveWrapper(BaseWrapper):
    """
    A wrapper over the ESSDeepDive API.

    This is a dynamic wrapper: it can be used as a search facade,
    but cannot be ingested in whole.
    """

    name: ClassVar[str] = "ess_deepdive"

    default_object_type = "Class"

    session: requests_cache.CachedSession = field(
        default_factory=lambda: requests_cache.CachedSession("ess_deepdive")
    )

    limit: int = field(default=50)

    def external_search(self, text: str, expand: bool = False, **kwargs) -> List:

        if expand:

            def qt(t: str):
                t = t.strip()
                if " " in t:
                    return f'"{t}"'
                return t

            logger.info(f"Expanding search term: {text} to create ESS-DeepDive query")
            model = self.extractor.model
            response = model.prompt(
                text,
                system="""
                Take the specified search text, and expand it to a list
                of key terms used to construct a query. You will return results as
                semi-colon separated list of the most relevant terms. Make sure to
                include all relevant concepts in the returned terms. For example,
                if the search term is 'I want data about soil temperature and water turbidity',
                you may return 'soil temperature; water turbidity'.""",
            )
            terms = response.text().split(";")
            logger.info(f"Expanded terms: {terms}")
            terms = [qt(t) for t in terms]
            search_term = "|".join(terms)
        else:
            search_term = text
        logger.info(f"Constructed search term: {search_term}")

        # This will store all search results.
        all_data = []

        # This will store multiple datasets matching the query.
        datasets = []

        # Parameters for the request.
        # This will search field names first,
        # then field definitions,
        # then field values (just text, not numeric).
        # It will save all results to all_data
        for search_field in ["fieldName", "fieldDefinition", "fieldValueText"]:
            params = {
                "rowStart": 1,
                "pageSize": 25,
                search_field: search_term,
            }
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            all_data.append(data)

        # Get all fields named "results" from all_data and combine them
        search_results = []
        for results_set in all_data:
            if "results" in results_set:
                search_results.extend(results_set["results"])

        snippets = {
            result["data_file_url"]: {
                "field_name": result["field_name"],
                "definition": result["definition"],
                "data_type": result["data_type"],
            }
            for result in search_results
        }
        ids = list(snippets.keys())

        for ident in ids:
            ident = ident.replace(BASE_URL + "/", "")
            # Yes, the doi defined here does not include the prefix
            # But the API will want it there
            doi, file_path = ident.split(":", 2)[1:]
            dataset_params = {
                "doi": f"doi:{doi}",
                "file_path": file_path,
            }

            dataset_response = requests.get(BASE_URL, params=dataset_params)
            if not dataset_response.ok:
                raise ValueError(f"Could not get dataset details for {ident}")
            dataset = dataset_response.json()

            # Extract and print this information
            if "fields" not in dataset:
                logger.error(f"Could not get pages from {ident}")
                this_data = {ident: snippets[ident]}
            else:
                logger.info(f"Got full field list for {ident}")
                this_data = {ident: (snippets[ident], dataset["fields"])}
            datasets.append(this_data)

        return datasets

    def objects(
        self, collection: str = None, object_ids: Optional[Iterable[str]] = None, **kwargs
    ) -> Iterator[Dict]:
        seen = set()
        for record in get_records(self.session, limit=self.limit):
            id = record["data_file_url"] + "-" + record["field_name"]
            if id in seen:
                logger.warning(f"Skipping duplicate record {id}")
                continue
            seen.add(id)
            record["id"] = id
            yield record
