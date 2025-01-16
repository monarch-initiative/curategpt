"""
This is a DuckDB adapter for the Vector Similarity Search (VSS) extension
using the experimental persistence feature
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
)

import duckdb
import llm
import numpy as np
import openai
import psutil
import yaml
from linkml_runtime.dumpers import json_dumper
from linkml_runtime.utils.yamlutils import YAMLRoot
from oaklib.utilities.iterator_utils import chunk
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from venomx.model.venomx import Index, Model

from curategpt.store.db_adapter import DBAdapter
from curategpt.store.duckdb_connection_handler import DuckDBConnectionAndRecoveryHandler
from curategpt.store.duckdb_result import DuckDBSearchResult
from curategpt.store.metadata import Metadata
from curategpt.store.vocab import (
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    DISTANCES,
    DOCUMENTS,
    EMBEDDINGS,
    IDS,
    METADATAS,
    MODEL_MAP,
    OBJECT,
    PROJECTION,
    QUERY,
    SEARCH_RESULT,
)
from curategpt.utils.vector_algorithms import mmr_diversified_search

logger = logging.getLogger(__name__)


@dataclass
class DuckDBAdapter(DBAdapter):
    name: ClassVar[str] = "duckdb"
    default_model: str = "all-MiniLM-L6-v2"
    conn: duckdb.DuckDBPyConnection = field(init=False)
    vec_dimension: int = field(init=False)
    ef_construction: int = 128
    ef_search: int = 64
    M: int = 16
    distance_metric: str = "cosine"
    id_field: str = "id"
    text_lookup: Optional[Union[str, Callable]] = field(default="text")
    id_to_object: Mapping[str, dict] = field(default_factory=dict)
    default_max_document_length: ClassVar[int] = 6000
    openai_client: OpenAI = field(default=None)

    def __post_init__(self):
        self.connection_handler = DuckDBConnectionAndRecoveryHandler(self.path)
        self.ef_construction = self._validate_ef_construction(self.ef_construction)
        self.ef_search = self._validate_ef_search(self.ef_search)
        self.M = self._validate_m(self.M)
        self.conn = self.connection_handler.connect()
        self.model = self.default_model
        self.vec_dimension = self._get_embedding_dimension(self.default_model)

    def _initialize_openai_client(self):
        if self.openai_client is None:
            from dotenv import load_dotenv

            load_dotenv()
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
            else:
                raise openai.OpenAIError(
                    "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
                )

    def _get_collection_name(self, collection: Optional[str] = None) -> str:
        """
        Get the collection name or the default collection name
        :param collection:
        :return:
        """
        return self._get_collection(collection)

    def _create_table_if_not_exists(
        self, collection: str, vec_dimension: int, venomx: Metadata = None
    ):
        """
        Create a table for the given collection if it does not exist
        :param collection:
        :return:
        """
        safe_collection_name = f'"{collection}"'
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {safe_collection_name} (
                id VARCHAR PRIMARY KEY,
                metadata JSON,
                embeddings FLOAT[{vec_dimension}],
                documents TEXT
            )
        """
        self.conn.execute(create_table_sql)

        if venomx:
            venomx = venomx.model_dump(exclude_none=True)
            # venomx metadata insertion
            self.conn.execute(
                f"""
                INSERT INTO {safe_collection_name} (id, metadata) VALUES ('__venomx__', ?)
                ON CONFLICT (id) DO NOTHING
                """,
                [venomx],
            )

    def create_index(self, collection: str):
        """
        Create an index for the given collection
        Parameters
        ----------
        collection

        Returns
        -------

        """
        cm = self.collection_metadata(collection)
        safe_collection_name = f'"{collection}"'
        index_name = f"{collection}_index"
        create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS "{index_name}" ON {safe_collection_name}
            USING HNSW (embeddings) WITH (
                metric='{cm.hnsw_space}',
                ef_construction={self.ef_construction},
                ef_search={self.ef_search},
                M={self.M}
            )
        """
        self.conn.execute(create_index_sql)

    def _embedding_function(
        self, texts: Union[str, List[str], List[List[str]]], model: str = None
    ) -> list:
        """
        Get the embeddings for the given texts using the specified model
        :param texts: A single text or a list of texts to embed
        :param model: Model to use for embedding
        :return: A single embedding or a list of embeddings
        """
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True

        if model is None:
            model = self.model

        if model.startswith("openai:"):
            self._initialize_openai_client()
            openai_model = model.split(":", 1)[1]
            if openai_model == "" or openai_model not in MODEL_MAP.keys():
                logger.info(
                    f"The model {openai_model} is not "
                    f"one of {[MODEL_MAP.keys()]}. Defaulting to {DEFAULT_OPENAI_MODEL}"
                )
                openai_model = DEFAULT_OPENAI_MODEL

            responses = [
                self.openai_client.embeddings.create(input=text, model=openai_model)
                .data[0]
                .embedding
                for text in texts
            ]
            return responses[0] if single_text else responses

        model = SentenceTransformer(model)
        embeddings = model.encode(texts, convert_to_tensor=False).tolist()
        return embeddings[0] if single_text else embeddings

    def insert(self, objs: Union[OBJECT, Iterable[OBJECT]], **kwargs):
        """
        Insert objects into the collection
        :param objs:
        :param kwargs:
        :return:
        """
        self._process_objects(objs, method="insert", **kwargs)

    # DELETE first to ensure primary key  constraint https://duckdb.org/docs/sql/indexes
    def update(self, objs: Union[OBJECT, Iterable[OBJECT]], **kwargs):
        """
        Update objects in the collection.
        :param objs:
        :param kwargs:
        :return:
        """
        collection = kwargs.get("collection")
        ids = [self._id(o, self.id_field) for o in objs]
        safe_collection_name = f'"{collection}"'
        delete_sql = f"DELETE FROM {safe_collection_name} WHERE id = ?"
        self.conn.executemany(delete_sql, [(id_,) for id_ in ids])
        self.insert(objs, **kwargs)

    def upsert(self, objs: Union[OBJECT, Iterable[OBJECT]], **kwargs):
        """
        Upsert objects into the collection
        :param objs:
        :param kwargs:
        :return:
        """
        collection = kwargs.get("collection")
        if collection not in self.list_collection_names():
            vec_dimension = self._get_embedding_dimension(kwargs.get("model"))
            self._create_table_if_not_exists(
                collection, vec_dimension, model=kwargs.get("model"), distance=self.distance_metric
            )
        ids = [self._id(o, self.id_field) for o in objs]
        existing_ids = set()
        for id_ in ids:
            safe_collection_name = f'"{collection}"'
            result = self.conn.execute(
                f"SELECT id FROM {safe_collection_name} WHERE id = ?", [id_]
            ).fetchall()
            if result:
                existing_ids.add(id_)
        objs_to_update = [o for o in objs if self._id(o, self.id_field) in existing_ids]
        objs_to_insert = [o for o in objs if self._id(o, self.id_field) not in existing_ids]
        if objs_to_update:
            self.update(objs_to_update, **kwargs)

        if objs_to_insert:
            self.insert(objs_to_insert, **kwargs)

    def _process_objects(
        self,
        objs: Union[OBJECT, Iterable[OBJECT]],
        collection: str = None,
        batch_size: int = None,
        object_type: str = None,
        model: str = None,
        distance: str = None,
        text_field: Union[str, Callable] = None,
        venomx: Optional[Metadata] = None,
        method: str = "insert",
        **kwargs,
    ):
        """
        Process objects by inserting, updating or upserting them into the collection
        :param objs:
        :param collection:
        :param batch_size:
        :param object_type:
        :param model:
        :param text_field:
        :param method:
        :param kwargs:
        :return:
        """
        collection = self._get_collection_name(collection)
        if model is None:
            model = self.default_model
        self.vec_dimension = self._get_embedding_dimension(model)

        updated_venomx = self.update_or_create_venomx(
            venomx,
            collection,
            model,
            distance,
            object_type,
            self.vec_dimension,
        )

        if collection not in self.list_collection_names():
            self._create_table_if_not_exists(
                collection,
                self.vec_dimension,
                venomx=updated_venomx,
            )

        # if collection already exists, update metadata here
        cm = self.update_collection_metadata(collection=collection, updated_venomx=updated_venomx)
        if isinstance(objs, Iterable) and not isinstance(objs, str):
            objs = list(objs)
        else:
            objs = [objs]
        obj_count = len(objs)
        kwargs.update({"object_count": obj_count})
        if batch_size is None:
            batch_size = 100000
        if text_field is None:
            text_field = self.text_lookup
        id_field = self.id_field
        sql_command = self._generate_sql_command(collection, method)
        sql_command = sql_command.format(collection=collection)
        if not self._is_openai(collection):
            for next_objs in chunk(objs, batch_size):
                next_objs = list(next_objs)
                docs = [self._text(o, text_field) for o in next_objs]
                metadatas = [self._dict(o) for o in next_objs]
                ids = [self._id(o, id_field) for o in next_objs]
                embeddings = self._embedding_function(docs, cm.venomx.embedding_model.name)
                try:
                    self.conn.execute("BEGIN TRANSACTION;")
                    self.conn.executemany(
                        sql_command, list(zip(ids, metadatas, embeddings, docs))  # noqa: B905
                    )
                    self.conn.execute("COMMIT;")
                except Exception as e:
                    self.conn.execute("ROLLBACK;")
                    logger.error(
                        f"Transaction failed: {e}, default model: {self.default_model}, model used: {model}, len(embeddings): {len(embeddings[0])}"
                    )
                    raise
                finally:
                    self.create_index(collection)
        else:
            if model.startswith("openai:"):
                openai_model = model.split(":", 1)[1]
                if openai_model == "" or openai_model not in MODEL_MAP.keys():
                    logger.info(
                        f"The model {openai_model} is not "
                        f"one of {MODEL_MAP.keys()}. Defaulting to {DEFAULT_OPENAI_MODEL}"
                    )
                    openai_model = DEFAULT_OPENAI_MODEL  # ada 002
                else:
                    logger.error(f"Something went wonky ## model: {model}")
            from transformers import GPT2Tokenizer

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            for next_objs in chunk(objs, batch_size):
                next_objs = list(next_objs)
                docs = [self._text(o, text_field) for o in next_objs]
                metadatas = [self._dict(o) for o in next_objs]
                ids = [self._id(o, id_field) for o in next_objs]

                tokenized_docs = [tokenizer.encode(doc) for doc in docs]
                current_batch = []
                current_token_count = 0
                batch_embeddings = []

                i = 0
                while i < len(tokenized_docs):
                    doc_tokens = tokenized_docs[i]
                    # peek
                    if current_token_count + len(doc_tokens) <= 8192:
                        current_batch.append(doc_tokens)
                        current_token_count += len(doc_tokens)
                        i += 1
                    else:
                        if current_batch:
                            logger.info(f"Tokens: {current_token_count}")
                            texts = [tokenizer.decode(tokens) for tokens in current_batch]
                            short_name, _ = MODEL_MAP[openai_model]
                            embedding_model = llm.get_embedding_model(short_name)
                            embeddings = list(embedding_model.embed_multi(texts, len(texts)))
                            logger.info(f"Number of Documents in batch: {len(embeddings)}")
                            batch_embeddings.extend(embeddings)

                        if len(doc_tokens) > 8192:
                            logger.warning(
                                f"Document with ID {ids[i]} exceeds the token limit alone and will be skipped."
                            )
                            # should not be happening as batched above
                            i += 1
                            continue
                        else:
                            current_batch = []
                            current_token_count = 0

                if current_batch:
                    logger.info(f"Last batch, token count: {current_token_count}")
                    texts = [tokenizer.decode(tokens) for tokens in current_batch]
                    short_name, _ = MODEL_MAP[openai_model]
                    embedding_model = llm.get_embedding_model(short_name)
                    embeddings = list(embedding_model.embed_multi(texts))
                    batch_embeddings.extend(embeddings)
                try:
                    self.conn.execute("BEGIN TRANSACTION;")
                    self.conn.executemany(
                        sql_command, list(zip(ids, metadatas, batch_embeddings, docs, strict=False))
                    )
                    self.conn.execute("COMMIT;")
                except Exception as e:
                    self.conn.execute("ROLLBACK;")
                    logger.error(
                        f"Transaction failed: {e}, default model: {self.default_model}, model used: {model}, len(embeddings): {len(embeddings[0])}"
                    )
                    raise
                finally:
                    # TODO: move outside - check memory/time profile
                    self.create_index(collection)

    def insert_from_huggingface(
        self,
        objs: Union[OBJECT, Iterable[OBJECT]],
        collection: str = None,
        batch_size: int = None,
        text_field: Union[str, Callable] = None,
        venomx: Optional[Metadata] = None,
        object_type: Optional[str] = None,
        distance: Optional[str] = None,
        vec_dimension: Optional[int] = None,
        method: str = "insert",
        **kwargs,
    ):
        collection = self._get_collection(collection)
        model = None
        try:
            if venomx:
                hf_metadata_model = venomx.venomx.embedding_model.name
                # object_type = venomx.object_type
                distance = venomx.hnsw_space
                # vec_dimension = venomx.venomx.embedding_dimension
                if hf_metadata_model:
                    model = hf_metadata_model
                vec_dimension = self._get_embedding_dimension(model)

        except Exception as e:
            raise KeyError(
                f"Metadata from {collection} is not compatible with the current version of CurateGPT"
            ) from e

        updated_venomx = self.update_or_create_venomx(
            venomx.venomx,
            collection,
            model,
            distance,
            object_type,
            vec_dimension,
        )
        if collection not in self.list_collection_names():
            self._create_table_if_not_exists(
                collection,
                vec_dimension,
                venomx=updated_venomx,
            )
        updated_venomx.venomx.id = collection  # prevent name error
        self.set_collection_metadata(collection_name=collection, metadata=updated_venomx)
        if batch_size is None:
            batch_size = 100000

        if not isinstance(objs, list):
            objs = list(objs)

        obj_count = len(objs)
        kwargs.update({"object_count": obj_count})

        sql_command = self._generate_sql_command(collection, method)
        sql_command = sql_command.format(collection=collection)

        for next_objs in chunk(objs, batch_size):
            next_objs = list(next_objs)
            ids = [item["metadata"]["id"] for item in next_objs]
            metadatas = [self._dict(o) for o in next_objs]
            documents = [item["document"] for item in next_objs]
            embeddings = [
                (
                    item["embeddings"].tolist()
                    if isinstance(item["embeddings"], np.ndarray)
                    else item["embeddings"]
                )
                for item in next_objs
            ]
            try:
                self.conn.execute("BEGIN TRANSACTION;")
                self.conn.executemany(
                    sql_command, list(zip(ids, metadatas, embeddings, documents, strict=False))
                )
                self.conn.execute("COMMIT;")
            except Exception as e:
                self.conn.execute("ROLLBACK;")
                logger.error(
                    f"Transaction failed: {e}, default model: {self.default_model}, model used: {model}, len(embeddings): {len(embeddings[0])}"
                )
                raise
            finally:
                self.create_index(collection)

    def update_or_create_venomx(
        self,
        venomx: Optional[Index],
        collection: str,
        model: str,
        distance: str,
        object_type: str,
        embeddings_dimension: Optional[int],
    ) -> Metadata:
        """
        Updates an existing Index instance (venomx) with additional values or creates a new one if none is provided.
        """
        # If venomx already exists, update its nested fields (as e.g. vec_dimension would not be given)
        if venomx:
            new_embedding_model = Model(name=model)
            updated_index = venomx.model_copy(
                update={  # given venomx comes as venomx=Index()
                    "embedding_model": new_embedding_model,
                    "embeddings_dimensions": embeddings_dimension,
                }
            )

            venomx = Metadata(venomx=updated_index, hnsw_space=distance, object_type=object_type)

        else:
            if distance is None:
                distance = self.distance_metric
            venomx = self.populate_venomx(
                collection, model, distance, object_type, embeddings_dimension
            )

        return venomx

    @staticmethod
    def populate_venomx(
        collection: Optional[str],
        model: Optional[str],
        distance: str,
        object_type: str,
        embeddings_dimension: int,
    ) -> Metadata:
        """
        Populate venomx with data currently given when inserting

        :param collection:
        :param model:
        :param distance:
        :param object_type:
        :param embeddings_dimension:
        :return:
        """
        venomx = Metadata(
            venomx=Index(
                id=collection,
                embedding_model=Model(name=model),
                embeddings_dimensions=embeddings_dimension,
            ),
            hnsw_space=distance,
            object_type=object_type,
        )
        return venomx

    def remove_collection(self, collection: str = None, exists_ok=False, **kwargs):
        """
        Remove the collection from the database
        :param collection:
        :param exists_ok:
        :param kwargs:
        :return:
        """
        collection = self._get_collection(collection)
        if not exists_ok:
            if collection not in self.list_collection_names():
                raise ValueError(f"Collection {collection} does not exist")
        # duckdb, requires that identifiers containing special characters ("-") must be enclosed in double quotes.
        safe_collection_name = f'"{collection}"'
        self.conn.execute(f"DROP TABLE IF EXISTS {safe_collection_name}")

    def search(
        self,
        text: str,
        where: QUERY = None,
        collection: str = None,
        limit: int = 10,
        relevance_factor: float = None,
        include=None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        """
        Search for objects in the collection that match the given text
        :param text:
        :param where:
        :param collection:
        :param limit:
        :param relevance_factor:
        :param include:
        :param kwargs:
        :return:
        """
        yield from self._search(
            text=text,
            where=where,
            collection=collection,
            limit=limit,
            relevance_factor=relevance_factor,
            include=include,
            **kwargs,
        )

    def _search(
        self,
        text: str,
        where: QUERY = None,
        collection: str = None,
        limit: int = 10,
        relevance_factor: float = None,
        model: str = None,
        include=None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        if relevance_factor is not None and relevance_factor < 1.0:
            yield from self._diversified_search(
                text=text,
                where=where,
                collection=collection,
                limit=limit,
                include=include,
                relevance_factor=relevance_factor,
                **kwargs,
            )
            return
        if include is None:
            include = {METADATAS, DOCUMENTS, DISTANCES}
        else:
            include = set(include)
        collection = self._get_collection(collection)
        cm = self.collection_metadata(collection)
        logger.info(f"Collection metadata={cm}")
        if model is None:
            if cm:
                model = cm.venomx.embedding_model.name
            if model is None:
                model = self.default_model
        logger.info(f"Model={model}")
        where_conditions = []
        if where:
            where_conditions.append(where)
        where_clause = " AND ".join(where_conditions)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        if relevance_factor is not None and relevance_factor < 1.0:
            yield from self._diversified_search(
                text, where, collection, limit, relevance_factor, include, **kwargs
            )
            return
        query_embedding = self._embedding_function(text, model)
        safe_collection_name = f'"{collection}"'

        vec_dimension = self._get_embedding_dimension(model)

        # TODO: !VERY IMPORTANT! distance metrics between Chroma and DuckDB have very different, unclear implementations
        # https://duckdb.org/docs/sql/functions/array.html#array_distancearray1-array2
        # https://docs.trychroma.com/guides
        # chromaDB: by default l2, other options are ip and cosine
        # duckDB: by default none, array_distance() or 1-array_cosine_similarity(), both bring different distances
        # than chromaDBs distance metric
        results = self.conn.execute(
            f"""
            SELECT *, array_distance(embeddings::FLOAT[{vec_dimension}],
            {query_embedding}::FLOAT[{vec_dimension}]) as distance
            FROM {safe_collection_name}
            {where_clause}
            ORDER BY distance
            LIMIT ?
        """,
            [limit],
        ).fetchall()
        yield from self.parse_duckdb_result(results, include)

    def _diversified_search(
        self,
        text: str,
        where: QUERY = None,
        collection: str = None,
        limit: int = 10,
        relevance_factor: float = 0.5,
        include=None,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        if limit is None:
            limit = 10
        # we need to set this as we need EMBEDDINGS
        include = {METADATAS, DOCUMENTS, EMBEDDINGS, DISTANCES}
        collection = self._get_collection(collection)
        cm = self.collection_metadata(collection)
        where_conditions = []
        if where:
            for key, value in where.items():
                where_conditions.append(f"json_extract(metadata, '$.{key}') = '{value}'")
        where_clause = " AND ".join(where_conditions)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        query_embedding = self._embedding_function(text, model=cm.venomx.embedding_model.name)
        safe_collection_name = f'"{collection}"'
        vec_dimension = self._get_embedding_dimension(cm.venomx.embedding_model.name)
        results = self.conn.execute(
            f"""
                    SELECT *, array_distance(embeddings::FLOAT[{vec_dimension}],
                    {query_embedding}::FLOAT[{vec_dimension}]) as distance
                    FROM {safe_collection_name}
                    {where_clause}
                    ORDER BY distance
                    LIMIT ?
                """,
            [limit * 10],
        ).fetchall()
        results = list(self.parse_duckdb_result(results, include))
        if not results:
            return
        rows = [np.array(r[2]["_embeddings"]) for r in results]
        query = np.array(query_embedding)
        reranked_indices = mmr_diversified_search(
            query, rows, relevance_factor=relevance_factor, top_n=limit
        )
        for i in reranked_indices:
            yield results[i]

    def list_collection_names(self):
        """
        List the names of all collections in the database
        :return:
        """
        result = self.conn.execute("PRAGMA show_tables;").fetchall()
        return [row[0] for row in result]

    def collection_metadata(
        self, collection_name: Optional[str] = None, include_derived=False, **kwargs
    ) -> Optional[Metadata]:
        """
        Get the metadata for the collection
        :param collection_name:
        :param include_derived:
        :param kwargs:
        :return:
        """
        collection_name = self._get_collection(collection_name)
        safe_collection_name = f'"{collection_name}"'
        try:
            result = self.conn.execute(
                f"SELECT metadata FROM {safe_collection_name} WHERE id = '__venomx__'"
            ).fetchone()
            if result:
                metadata = json.loads(result[0])
                metadata = Metadata(**metadata)
                # metadata = result[0]
                if include_derived:
                    # not implemented yet
                    # metadata_instance.object_count = compute_object_count(collection_name
                    pass
                return metadata
        except Exception as e:
            logger.error(f"Failed to retrieve metadata for collection {collection_name}: {str(e)}")
            return None

    def update_collection_metadata(self, collection: str, **kwargs):
        """
        Update the metadata for a collection. This function will merge new metadata provided
        via kwargs with existing metadata, if any, ensuring that only the specified fields
        are updated.
        :param collection:
        :param kwargs:
        :return:
        """

        if not collection:
            raise ValueError("Collection name must be provided.")
        metadata = self.collection_metadata(collection)
        current_venomx = {**kwargs}
        if metadata is None:  # should not be possible
            logger.warning(
                f"No existing metadata found for collection {collection}. Initializing new metadata."
            )
            metadata = Metadata(venomx=Index(**current_venomx))
        else:
            metadata_dict = metadata.model_dump(exclude_none=True)
            # Check if the existing venomx has an embedding model and if it matches the one in kwargs
            if "venomx" in metadata_dict and metadata_dict["venomx"].get("embedding_model"):
                existing_model_name = metadata_dict["venomx"]["embedding_model"].get("name")
                new_model_name = current_venomx.get("embedding_model", {}).get("name")

                if new_model_name and existing_model_name and new_model_name != existing_model_name:
                    raise ValueError(
                        f"Cannot change the embedding model name from '{existing_model_name}' to '{new_model_name}'. "
                        f"Model dimensions are incompatible with changes to the model."
                    )

        # Merge current_venomx (from kwargs) into the nested venomx dictionary
        if "venomx" in metadata_dict and isinstance(metadata_dict["venomx"], dict):
            metadata_dict["venomx"].update(current_venomx)
        else:
            metadata_dict["venomx"] = current_venomx
            # Reconstruct the Metadata object from the updated dictionary
            metadata = Metadata(**metadata_dict)
        updated_metadata_dict = metadata.model_dump(exclude_none=True)

        safe_collection_name = f'"{collection}"'
        self.conn.execute(
            f"""
                UPDATE {safe_collection_name} SET metadata = ?
                WHERE id = '__venomx__'
                """,
            [updated_metadata_dict],
        )
        return metadata

    def set_collection_metadata(self, collection_name: Optional[str], metadata: Metadata, **kwargs):
        """
        Set the metadata for the collection
        :param collection_name:
        :param metadata:
        :param kwargs:
        :return:
        """
        if collection_name is None:
            raise ValueError("Collection name must be provided.")

        current_metadata = self.collection_metadata(collection_name)

        if metadata:
            if metadata.venomx.id != collection_name:
                raise ValueError(
                    f"venomx.id: {metadata.venomx.id} must match collection_name {collection_name}"
                )

            new_model = metadata.venomx.embedding_model.name

            prev_model = current_metadata.venomx.embedding_model.name
            if prev_model and new_model != prev_model:
                raise ValueError(f"Cannot change model from {prev_model} to {new_model}")

        # metadata_json = json.dumps(metadata.dict(exclude_none=True))
        metadata = metadata.model_dump(exclude_none=True)
        safe_collection_name = f'"{collection_name}"'
        self.conn.execute(
            f"""
            UPDATE {safe_collection_name}
            SET metadata = ?
            WHERE id = '__venomx__'
            """,
            [metadata],
        )

    def find(
        self,
        where: QUERY = None,
        projection: PROJECTION = None,
        collection: str = None,
        include=None,
        limit: int = 10,
        **kwargs,
    ) -> Iterator[SEARCH_RESULT]:
        """
        Find objects in the collection that match the given query and projection

        :param where: the query to filter the results
        :param projection:
        :param collection: name of the collection to search
        :param include: fields to be included in output
        :param limit: maximum number of results to return
        :param kwargs:
        :return:

        Parameters
        ----------
        """
        collection = self._get_collection(collection)
        where_clause = self._parse_where_clause(where) if where else ""
        where_clause = f"WHERE {where_clause}" if where_clause else ""
        if include is None:
            include = [IDS, METADATAS, DOCUMENTS]
        safe_collection_name = f'"{collection}"'
        query = f"""
                    SELECT id, metadata, embeddings, documents, NULL as distance
                    FROM {safe_collection_name}
                    {where_clause}
                    LIMIT {limit}
                """
        results = self.conn.execute(query).fetchall()
        yield from self.parse_duckdb_result(results, include)

    def matches(self, obj: OBJECT, include=None, **kwargs) -> Iterator[SEARCH_RESULT]:
        """
        Find objects in the collection that match the given object
        :param obj:
        :param include:
        :param kwargs:
        :return:
        """
        if include is None:
            include = {IDS, METADATAS, DOCUMENTS, DISTANCES}
        else:
            include = set(include)
        text_field = self.text_lookup
        logger.info(f"## TEXT FIELD:{text_field}")
        text = self._text(obj, text_field)
        logger.info(f"{text}")
        logger.info(f"Query term: {text}")
        yield from self.search(text=text, include=include, **kwargs)

    def lookup(self, id: str, collection: str = None, include=None, **kwargs) -> OBJECT:
        """
        Lookup an object by its id
        :param id: ID of the object to lookup
        :param collection: Name of the collection to search
        :param include: List of fields to include in the output ['metadata', 'embeddings', 'documents']
        :param kwargs:
        :return:
        """
        if include is None:
            include = {METADATAS}
        else:
            include = set(include)
        safe_collection_name = f'"{collection}"'
        result = self.conn.execute(
            f"""
                SELECT *
                FROM {safe_collection_name}
                WHERE id = ?
            """,
            [id],
        ).fetchone()
        if isinstance(result, tuple) and len(result) > 1:
            search_result = DuckDBSearchResult(
                ids=result[0],
                metadatas=json.loads(result[1]),
                embeddings=result[2],
                documents=result[3],
                include=include,
            )
            return search_result.to_dict().get(METADATAS)

    def peek(
        self, collection: str = None, limit=5, include=None, offset: int = 0, **kwargs
    ) -> Iterator[SEARCH_RESULT]:
        """
        Peek at the first N objects in the collection
        :param collection:
        :param limit:
        :param include:
        :param offset:
        :param kwargs:
        :return:
        """
        if include is None:
            include = {IDS, METADATAS, DOCUMENTS}
        else:
            include = set(include)
        safe_collection_name = f'"{collection}"'
        results = self.conn.execute(
            f"""
                SELECT id, metadata, embeddings, documents, NULL as distance
                FROM {safe_collection_name}
                LIMIT ?
            """,
            [limit],
        ).fetchall()

        yield from self.parse_duckdb_result(results, include)

    def fetch_all_objects_memory_safe(
        self, collection: str = None, batch_size: int = 100, include=None, **kwargs
    ) -> Iterator[OBJECT]:
        """
        Fetch all objects from a collection, in batches to avoid memory overload.
        """
        collection = self._get_collection(collection)
        offset = 0
        while True:
            if include is None:
                include = [IDS, METADATAS, DOCUMENTS, EMBEDDINGS]
            safe_collection_name = f'"{collection}"'
            query = f"""
                                SELECT id, metadata, embeddings, documents, NULL as distance
                                FROM {safe_collection_name}
                                LIMIT ? OFFSET ?
                            """
            results = self.conn.execute(query, [batch_size, offset]).fetchall()
            if results:
                yield from self.parse_duckdb_result(results, include)
                offset += batch_size
            else:
                break

    def get_raw_objects(self, collection) -> Iterator[Dict]:
        """
        Get all raw objects in the collection as they were inserted into the database
        :param collection:
        :return:
        """
        safe_collection_name = f'"{collection}"'
        results = self.conn.execute(
            f"""
                SELECT metadata
                FROM {safe_collection_name}
            """
        ).fetchall()
        for result in results:
            yield json.loads(result[0])

    def dump_then_load(
        self,
        collection: str = None,
        target: DBAdapter = None,
    ):
        """
        Dump the collection to a file and then load it into the target adapter
        :param collection:
        :param target:
        :param temp_file:
        :param format:
        :return:
        """
        if collection is None:
            raise ValueError("Collection name must be provided.")
        if not isinstance(target, DuckDBAdapter):
            raise ValueError("Target must be a DuckDBAdapter instance")

        result = self.get_raw_objects(collection)

        metadata = self.collection_metadata(collection)
        model = metadata["model"]
        vec_dimension = self._get_embedding_dimension(model)
        distance = metadata["hnsw_space"]
        # in case it exists already, remove
        target.remove_collection(collection, exists_ok=True)
        # using same collection name in target database
        target._create_table_if_not_exists(collection, vec_dimension, distance, model)
        target.set_collection_metadata(collection, metadata)
        batch_size = 5000
        for i in range(0, len(list(result)), batch_size):
            batch = result[i : i + batch_size]
            target.insert(batch, collection=collection)

    @staticmethod
    def kill_process(pid):
        """
        Kill the process with the given PID
        Returns
        -------

        """
        process = None
        try:
            process = psutil.Process(pid)
            process.terminate()  # Sends SIGTERM
            process.wait(timeout=5)
        except psutil.NoSuchProcess:
            logger.info("Process already terminated.")
        except psutil.TimeoutExpired:
            if process is not None:
                logger.warning("Process did not terminate in time, forcing kill.")
                process.kill()  # Sends SIGKILL as a last resort
        except Exception as e:
            logger.error(f"Failed to terminate process: {e}")

    @staticmethod
    def _generate_sql_command(collection: str, method: str) -> str:
        safe_collection_name = f'"{collection}"'
        if method == "insert":
            return f"""
                INSERT INTO {safe_collection_name} (id,metadata, embeddings, documents) VALUES (?, ?, ?, ?)
                """
        else:
            raise ValueError(f"Unknown method: {method}")

    def _is_openai(self, collection: str) -> bool:
        """
        Check if the collection uses a OpenAI Embedding model
        :param collection:
        :return:
        """
        collection = self._get_collection(collection)
        safe_collection_name = f'"{collection}"'
        query = f"SELECT metadata FROM {safe_collection_name} WHERE id = '__metadata__'"
        result = self.conn.execute(query).fetchone()
        if result:
            metadata = json.loads(result[0])
            if "model" in metadata and metadata["model"].startswith("openai:"):
                return True
        return False

    def _dict(self, obj: OBJECT):
        if isinstance(obj, dict):
            return obj
        elif isinstance(obj, BaseModel):
            return obj.model_dump(exclude_unset=True)
        elif isinstance(obj, YAMLRoot):
            return json_dumper.to_dict(obj)
        else:
            raise ValueError(f"Cannot convert {obj} to dict")

    def _id(self, obj, id_field):
        if isinstance(obj, dict):
            id = obj.get(id_field, None)
        else:
            id = getattr(obj, id_field, None)
        if not id:
            id = str(obj)
        self.id_to_object[id] = obj
        return id

    def _text(self, obj: OBJECT, text_field: Union[str, Callable]):
        if isinstance(obj, DuckDBSearchResult):
            obj = obj.dict()
        if isinstance(obj, list):
            raise ValueError(f"Cannot handle list of text fields: {obj}")
        if text_field is None or (isinstance(text_field, str) and text_field not in obj):
            obj = {k: v for k, v in obj.items() if v}
            t = yaml.safe_dump(obj, sort_keys=False)
        elif isinstance(text_field, Callable):
            t = text_field(obj)
        elif isinstance(obj, dict):
            t = obj[text_field]
        else:
            t = getattr(obj, text_field)
        t = t.strip()
        if not t:
            raise ValueError(f"Text field {text_field} is empty for {type(obj)} : {obj}")
        if len(t) > self.default_max_document_length:
            logger.warning(f"Truncating text field {text_field} for {str(obj)[0:100]}...")
            t = t[: self.default_max_document_length]
        return t

    def identifier_field(self, collection: str = None) -> str:
        if self.schema_proxy and self.schema_proxy.schemaview:
            fields = []
            for s in self.schema_proxy.schemaview.all_slots(attributes=True).values():
                if s.identifier:
                    fields.append(s.name)
            if fields:
                if len(fields) > 1:
                    raise ValueError(f"Multiple identifier fields: {fields}")
                return fields[0]
        return "id"

    @staticmethod
    def parse_duckdb_result(results, include) -> Iterator[SEARCH_RESULT]:
        """
        Parse the results from the SQL
        :return: DuckDBSearchResultIterator
        ----------
        """
        for res in results:
            if res[0] != "__metadata__" and res[0] != "__venomx__":
                D = DuckDBSearchResult(
                    ids=res[0],
                    metadatas=json.loads(res[1]),
                    embeddings=res[2],
                    documents=res[3],
                    distances=res[4],
                    include=include,
                )
                yield from D.__iter__()

    @staticmethod
    def _parse_where_clause(where: Dict[str, Any]) -> str:
        """
        Parse the where clause from the query
        Parameters
        ----------
        where

        Returns
        -------

        """
        conditions = []
        for key, condition in where.items():
            if isinstance(condition, dict):
                for op, value in condition.items():
                    if op == "$eq":
                        conditions.append(f"json_extract_string(metadata, '$.{key}') = '{value}'")
                    elif op == "$ne":
                        conditions.append(f"json_extract_string(metadata, '$.{key}') != '{value}'")
                    elif op == "$gt":
                        conditions.append(
                            f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) > '{value}'"
                        )
                    elif op == "$gte":
                        conditions.append(
                            f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) >= '{value}'"
                        )
                    elif op == "$lt":
                        conditions.append(
                            f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) < '{value}'"
                        )
                    elif op == "$lte":
                        conditions.append(
                            f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) <= '{value}'"
                        )
                    elif op == "$in":
                        conditions.append(
                            f"json_extract_string(metadata, '$.{key}') IN ({', '.join([f'{v}' for v in value])})"
                        )
                    elif op == "$nin":
                        conditions.append(
                            f"json_extract_string(metadata, '$.{key}') NOT IN ({', '.join([f'{v}' for v in value])})"
                        )
                    elif op == "$exists":
                        if value:
                            conditions.append(f"json_extract(metadata, '$.{key}') IS NOT NULL")
                        else:
                            conditions.append(f"json_extract(metadata, '$.{key}') IS NULL")
                    elif op == "$regex":
                        conditions.append(f"json_extract_string(metadata, '$.{key}') ~ '{value}'")
            else:
                conditions.append(f"json_extract_string(metadata, '$.{key}') = '{condition}'")
        return " AND ".join(conditions)

    def _get_embedding_dimension(self, model_name: str) -> int:
        if model_name is None or model_name.startswith(self.default_model):
            return DEFAULT_MODEL[self.default_model]
        if isinstance(model_name, str):
            if model_name.startswith("openai:"):
                model_key = model_name.split("openai:", 1)[1]
                if model_key == "" or model_key not in MODEL_MAP.keys():
                    model_key = DEFAULT_OPENAI_MODEL
                model_info = MODEL_MAP.get(model_key, DEFAULT_OPENAI_MODEL)
                return model_info[1]
            else:
                return MODEL_MAP[DEFAULT_OPENAI_MODEL][1]

    @staticmethod
    def _validate_ef_construction(value: int) -> int:
        """
        The number of candidate vertices to consider during the construction of the index. A higher value will result
        in a more accurate index, but will also increase the time it takes to build the index.
        Parameters
        ----------
        value

        Returns
        -------

        """
        if not (10 <= value <= 200):
            raise ValueError("ef_construction must be between 10 and 200")
        return value

    @staticmethod
    def _validate_ef_search(value: int) -> int:
        """
        The number of candidate vertices to consider during the search phase of the index.
        A higher value will result in a more accurate index, but will also increase the time it takes to perform a search.
        Parameters
        ----------
        value

        Returns
        -------

        """
        if not (10 <= value <= 200):
            raise ValueError("ef_search must be between 10 and 200")
        return value

    @staticmethod
    def _validate_m(value: int) -> int:
        """
        The maximum number of neighbors to keep for each vertex in the graph.
        A higher value will result in a more accurate index, but will also increase the time it takes to build the index.
        Parameters
        ----------
        value

        Returns
        -------

        """
        if not (5 <= value <= 48):
            raise ValueError("M must be between 5 and 48")
        return value

    @staticmethod
    def determine_fields_to_include(include: Optional[List[str]] = None) -> str:
        """
        Determine which fields to include in the SQL query based on the 'include' parameter.

        :param include: List of fields to include in the output ['metadata', 'embeddings', 'documents']
        :return: Comma-separated string of fields to include
        """
        fields = []
        if include is None or IDS in include:
            fields.append(IDS)
        if include is None or METADATAS in include:
            fields.append(METADATAS)
        if include is None or EMBEDDINGS in include:
            fields.append(EMBEDDINGS)
        if include is None or DOCUMENTS in include:
            fields.append(DOCUMENTS)
        return ", ".join(fields)
