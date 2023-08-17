"""Chat with a KB."""
import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Iterator, List, Optional, Union

from deprecation import deprecated

from curate_gpt import ChromaDBAdapter
from curate_gpt.extract import Extractor
from curate_gpt.store import DBAdapter
from curate_gpt.store.db_adapter import SEARCH_RESULT

logger = logging.getLogger(__name__)


@dataclass
class BaseWrapper(ABC):
    """
    A virtual store that implements a view over some remote or external source.
    """

    local_store: DBAdapter = None
    """Adapter to local knowledge base used to cache results."""

    extractor: Extractor = None

    source_locator: Optional[Union[str, Path]] = None

    name: ClassVar[str] = "__dbview__"

    default_embedding_model = "openai:"

    default_object_type = "Publication"

    def search(
        self,
        text: str,
        collection: str = None,
        cache: bool = True,
        expand: bool = True,
        external_search_limit: Optional[int] = None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        """
        Search an external source and cache the results in the local store.

        :param text:
        :param collection: used for caching
        :param kwargs:
        :return:
        """
        logger.info(f"Searching for {text}")
        ext_kwargs = {k: v for k, v in kwargs.items() if k != "limit"}
        parsed_data = self.external_search(
            text, expand=expand, limit=external_search_limit, **ext_kwargs
        )
        db = self.local_store
        if db is None:
            tmpdir = Path("/tmp")
            db = ChromaDBAdapter(tmpdir)
        if collection is None:
            collection = self._cached_collection_name(is_temp=not cache)
        if not cache:
            db.remove_collection(collection, exists_ok=True)
        logger.info(f"Inserting {len(parsed_data)} records into {collection}")
        db.upsert(parsed_data, collection=collection, model=self.default_embedding_model)
        db.update_collection_metadata(
            collection,
            object_type="Publication",
            description=f"Special cache for {self.name} searches",
        )
        yield from db.search(text, collection=collection, **kwargs)

    def objects(
        self, collection: str = None, object_ids: Iterable[str] = None, **kwargs
    ) -> Iterator[Dict]:
        """
        Return all objects in the view.

        :param collection:
        :param object_ids: Optional list of IDs to fetch
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def external_search(self, text: str, expand: bool = True, **kwargs) -> List:
        """
        Search an external source and return the results.

        :param text:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def extract_concepts_from_text(self, text: str, **kwargs):
        model = self.extractor.model
        response = model.prompt(
            text, system="generate a semi-colon separated list of the most relevant terms"
        )
        terms = response.text().split(";")
        return terms

    @deprecated("Use a chat agent with a wrapper as source instead")
    def chat(
        self,
        query: str,
        collection: str = None,
        **kwargs,
    ) -> Any:
        """
        Chat with pubmed.

        :param query:
        :param collection:
        :param kwargs:
        :return:
        """
        from curate_gpt.agents.chat_agent import ChatAgent

        # prime the pubmed cache
        if collection is None:
            collection = self._cached_collection_name()
        logger.info(f"Ensure pubmed cached for {query}, kwargs={kwargs}, self={self}")
        _ = list(self.search(query, collection=collection, **kwargs))
        # ensure the collection exists and is configured correctly
        self.local_store.update_collection_metadata(
            collection,
            model=self.default_embedding_model,
            object_type=self.default_object_type,
            description=f"Special cache for {self.name} searches",
        )
        chat = ChatAgent(knowledge_source=self.local_store, extractor=self.extractor)
        response = chat.chat(query, collection=collection)
        return response

    def _cached_collection_name(self, is_temp=False) -> str:
        if is_temp:
            return f"__{self.name}_temp__"
        else:
            return self.name + "_api_cached"
