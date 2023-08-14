"""ChromaDB adapter."""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import chromadb
import yaml
from chromadb import API, Settings
from chromadb.api import EmbeddingFunction
from chromadb.types import Collection
from chromadb.utils import embedding_functions
from linkml_runtime.dumpers import json_dumper
from linkml_runtime.utils.yamlutils import YAMLRoot
from pydantic import BaseModel

from curate_gpt.store.db_adapter import (
    DEFAULT_COLLECTION,
    OBJECT,
    PROJECTION,
    QUERY,
    SEARCH_RESULT,
    CollectionMetadata,
    DBAdapter,
)
from curate_gpt.utils.search import mmr_diversified_search

logger = logging.getLogger(__name__)


@dataclass
class ChromaDBAdapter(DBAdapter):
    """
    An Adapter that wraps a ChromaDB client
    """

    # model: str = field(default="all-MiniLM-L6-v2")
    default_model = "all-MiniLM-L6-v2"
    client: API = None
    id_field: str = field(default="id")
    text_lookup: Optional[Union[str, Callable]] = field(default="text")
    id_to_object: Mapping[str, OBJECT] = field(default_factory=dict)

    default_max_document_length: ClassVar[int] = 6000  # TODO: use tiktoken

    def __post_init__(self):
        if not self.path:
            self.path = "./db"
        logger.info(f"Using ChromaDB at {self.path}")
        self.client = chromadb.PersistentClient(path=self.path, settings=Settings(allow_reset=True))

    def _text(self, obj: OBJECT, text_field: Union[str, Callable]):
        if text_field is None or (isinstance(text_field, str) and text_field not in obj):
            obj = {k: v for k, v in obj.items() if v}
            t = yaml.safe_dump(obj, sort_keys=False)
        elif isinstance(text_field, Callable):
            t = text_field(obj)
        elif isinstance(obj, dict):
            t = obj[text_field]
        else:
            t = getattr(obj, text_field)
        if not t:
            raise ValueError(f"Text field {text_field} is empty for {obj}")
        if len(t) > self.default_max_document_length:
            logger.warning(f"Truncating text field {text_field} for {obj}")
            t = t[: self.default_max_document_length]
        return t

    def _id(self, obj: OBJECT, id_field: str):
        if isinstance(obj, dict):
            id = obj.get(id_field, None)
        else:
            id = getattr(obj, id_field, None)
        if not id:
            id = str(obj)
        self.id_to_object[id] = obj
        return id

    def _dict(self, obj: OBJECT):
        if isinstance(obj, dict):
            return obj
        elif isinstance(obj, BaseModel):
            return obj.dict(exclude_unset=True)
        elif isinstance(obj, YAMLRoot):
            return json_dumper.to_dict(obj)
        else:
            raise ValueError(f"Cannot convert {obj} to dict")

    def _object_metadata(self, obj: OBJECT):
        """
        Transform an object into metadata suitable for storage in chromadb

        :param obj:
        :return:
        """
        dict_obj = self._dict(obj)
        dict_obj["_json"] = json.dumps(dict_obj)
        return {
            k: v for k, v in dict_obj.items() if not isinstance(v, (dict, list)) and v is not None
        }

    def reset(self):
        """
        Reset the database.
        """
        self.client.reset()

    def _embedding_function(self, model: str = None) -> EmbeddingFunction:
        """
        Get the embedding function for a given model.

        :param model:
        :return:
        """
        if model is None:
            raise ValueError("Model must be specified")
        if model.startswith("openai:"):
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"), model_name="text-embedding-ada-002"
            )
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)

    def insert(
        self,
        objs: Union[OBJECT, Iterable[OBJECT]],
        **kwargs,
    ):
        self._insert_or_update(objs, method_name="add", **kwargs)

    def _insert_or_update(
        self,
        objs: Union[OBJECT, Iterable[OBJECT]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = None,
        method_name="add",
        object_type: str = None,
        model: str = None,
        text_field: Union[str, Callable] = None,
        **kwargs,
    ):
        """
        Insert an object or list of objects into the database.

        :param objs:
        :param collection:
        :param kwargs:
        :return:
        """
        client = self.client
        cm = self.collection_metadata(collection)
        if model is None:
            if cm:
                model = cm.model
            if model is None:
                model = self.default_model
        cm = self.update_collection_metadata(collection, model=model, object_type=object_type)
        ef = self._embedding_function(cm.model)
        # cm = CollectionMetadata(name=collection, model=self.model, object_type=object_type)
        cm_dict = cm.dict(exclude_none=True)
        collection_obj = client.get_or_create_collection(
            name=collection,
            embedding_function=ef,
            metadata=cm_dict,
        )
        if not isinstance(objs, list):
            if isinstance(objs, Iterable):
                # TODO: iterate in chunks
                objs = list(objs)
            else:
                objs = [objs]
        if self._is_openai(collection_obj) and batch_size is None:
            batch_size = 100
        if text_field is None:
            text_field = self.text_lookup
        id_field = self.identifier_field(collection)
        logger.info(f"Preparing texts...")
        i = 0
        while i < len(objs):
            logger.info(f"Preparing batch from position {i}...")
            # see https://github.com/chroma-core/chroma/issues/709
            if batch_size:
                next_objs = objs[i : i + batch_size]
                i += batch_size
            else:
                next_objs = objs
                i = len(objs)
            docs = [self._text(o, text_field) for o in next_objs]
            logger.debug(f"Example doc (tf={text_field}): {docs[0]}")
            logger.info(f"Preparing metadatas...")
            metadatas = [self._object_metadata(o) for o in next_objs]
            logger.info(f"Preparing ids...")
            ids = [self._id(o, id_field) for o in next_objs]
            logger.info(f"Inserting {len(next_objs)} / {len(objs)} objects into {collection}")
            method = getattr(collection_obj, method_name)
            method(
                documents=docs,
                metadatas=metadatas,
                ids=ids,
            )

    def update(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """
        Update an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        self._insert_or_update(objs, method_name="update", **kwargs)

    def upsert(self, objs: Union[OBJECT, List[OBJECT]], **kwargs):
        """
        Update an object or list of objects in the store.

        :param objs:
        :param collection:
        :return:
        """
        self._insert_or_update(objs, method_name="upsert", **kwargs)

    def remove_collection(self, collection: str = DEFAULT_COLLECTION, exists_ok=False, **kwargs):
        """
        Remove a collection from the database.

        :param collection:
        :param exists_ok:
        :return:
        """
        try:
            collection_obj = self.client.get_collection(name=collection)
        except Exception as e:
            if not exists_ok:
                raise e
            return
        collection_obj.delete()

    def _unjson(self, obj: Mapping):
        return json.loads(obj["_json"])

    def list_collection_names(self) -> List[str]:
        """
        List all collections in the database.

        :return:
        """
        return [c.name for c in self.client.list_collections()]

    def collection_metadata(
        self, collection_name: Optional[str] = DEFAULT_COLLECTION, include_derived=False, **kwargs
    ) -> Optional[CollectionMetadata]:
        """
        Get the metadata for a collection.

        :param collection_name:
        :return:
        """
        try:
            collection_obj = self.client.get_collection(name=collection_name)
        except Exception as e:
            return None
        cm = CollectionMetadata(**collection_obj.metadata)
        if include_derived:
            cm.object_count = collection_obj.count()
        return cm

    def set_collection_metadata(
        self, collection_name: Optional[str], metadata: CollectionMetadata, **kwargs
    ):
        """
        Set the metadata for a collection.

        :param collection_name:
        :param metadata:
        :return:
        """
        self.update_collection_metadata(
            collection_name=collection_name, **metadata.dict(exclude_none=True)
        )

    def update_collection_metadata(self, collection_name: str, **kwargs) -> CollectionMetadata:
        """
        Update the metadata for a collection.

        :param collection_name:
        :param kwargs:
        :return:
        """
        metadata = self.collection_metadata(collection_name=collection_name)
        if metadata is None:
            metadata = CollectionMetadata(**kwargs)
        else:
            prev_model = metadata.model
            metadata = metadata.copy(update=kwargs)
            if prev_model and metadata.model != prev_model:
                if self.client.get_or_create_collection(name=collection_name).count() > 0:
                    raise ValueError(f"Cannot change model from {prev_model} to {metadata.model}")
                else:
                    logger.info(
                        f"Changing (empy collection) model from {prev_model} to {metadata.model}"
                    )
        # self.set_collection_metadata(collection_name=collection_name, metadata=metadata)
        if metadata.name:
            assert metadata.name == collection_name
        else:
            metadata.name = collection_name
        self.client.get_or_create_collection(
            name=collection_name, metadata=metadata.dict(exclude_none=True)
        )
        return metadata

    def search(self, text: str, **kwargs) -> Iterator[SEARCH_RESULT]:
        yield from self._search(text=text, **kwargs)

    def _search(
        self,
        text: str = None,
        where: QUERY = None,
        collection: str = DEFAULT_COLLECTION,
        limit=10,
        include=None,
        relevance_factor: float = None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        logger.info(f"Searching for {text} in {collection}")
        if relevance_factor is not None and relevance_factor < 1.0:
            yield from self.diversified_search(
                text=text,
                where=where,
                collection=collection,
                limit=limit,
                relevance_factor=relevance_factor,
                **kwargs,
            )
            return
        if not include:
            include = ["metadatas", "documents", "distances"]
        if "*" in include:
            include = ["metadatas", "documents", "distances", "embeddings"]
        client = self.client
        collection = client.get_collection(name=collection)
        metadata = collection.metadata
        collection = client.get_collection(
            name=collection.name, embedding_function=self._embedding_function(metadata["model"])
        )
        logger.debug(f"Collection metadata: {metadata}")
        if text:
            query_texts = [text]
        else:
            query_texts = [""]
        results = collection.query(
            query_texts=query_texts, where=where, n_results=limit, include=include, **kwargs
        )
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        if "embeddings" in include:
            embeddings = results["embeddings"][0]
        else:
            embeddings = None
        for i in range(0, len(documents)):
            if embeddings:
                embeddings_i = embeddings[i]
            else:
                embeddings_i = None
            yield self._unjson(metadatas[i]), distances[i], {
                "embeddings": embeddings_i,
                "document": documents[i],
            }

    def diversified_search(
        self,
        text: str = None,
        limit: int = None,
        relevance_factor=0.5,
        collection: str = DEFAULT_COLLECTION,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        if limit is None:
            limit = 10
        logger.info(
            f"diversified search RF={relevance_factor}, text={text}, limit={limit}, collection={collection}"
        )
        collection_obj = self.client.get_collection(name=collection)
        metadata = collection_obj.metadata
        ef = self._embedding_function(metadata["model"])
        query_embedding = ef([text])
        kwargs["include"] = ["metadatas", "documents", "distances", "embeddings"]
        ranked_results = list(self.search(text, limit=limit * 10, collection=collection, **kwargs))
        if not ranked_results:
            return
        import numpy as np

        rows = [np.array(r[2]["embeddings"]) for r in ranked_results]
        query = np.array(query_embedding[0])
        reranked_indices = mmr_diversified_search(
            query, rows, relevance_factor=relevance_factor, top_n=limit
        )
        for i in reranked_indices:
            yield ranked_results[i]

    def find(
        self,
        where: QUERY,
        projection: PROJECTION = None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        yield from self.search("", where=where, **kwargs)

    def matches(self, obj: OBJECT, **kwargs) -> Iterator[SEARCH_RESULT]:
        """
        Query the database for matches to an object.

        :param obj:
        :param kwargs:
        :return:
        """
        text_field = self.text_lookup
        text = self._text(obj, text_field)
        logger.info(f"Query term: {text}")
        yield from self.search(text, **kwargs)

    def lookup(self, id: str, collection: str = DEFAULT_COLLECTION, **kwargs) -> OBJECT:
        """
        Lookup an object by its ID.

        :param id:
        :param collection:
        :return:
        """
        client = self.client
        collection = client.get_or_create_collection(name=collection)
        results = collection.get([id], include=["metadatas"])
        return self._unjson(results["metadatas"][0])

    def collections(self) -> Iterator[str]:
        """
        Return the names of all collections in the database.

        :return:
        """
        client = self.client
        for c in client.list_collections():
            yield c.name

    def _is_openai(self, collection: Collection):
        if collection.metadata.get("model", "").startswith("openai:"):
            return True

    def peek(self, collection: str = DEFAULT_COLLECTION, limit=5, **kwargs) -> Iterator[OBJECT]:
        c = self.client.get_collection(name=collection)
        logger.debug(f"Peeking at {collection}")
        results = c.peek(limit=limit)
        # TODO: DRY
        metadatas = results["metadatas"]
        for i in range(0, len(metadatas)):
            yield self._unjson(metadatas[i])
