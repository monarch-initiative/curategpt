"""
This is a DuckDB adapter for the Vector Similarity Search (VSS) extension
using the experimental persistence feature
"""
import psutil
import yaml
import logging
import os
import time
import re

import numpy as np
from dataclasses import dataclass, field
from typing import ClassVar, Iterable, Iterator, Optional, Union, Callable, Mapping, List, Dict, Any
import duckdb
import json

import openai
from openai import OpenAI
from pathlib import Path

from linkml_runtime.dumpers import json_dumper
from linkml_runtime.utils.yamlutils import YAMLRoot
from oaklib.utilities.iterator_utils import chunk
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from curate_gpt.store.db_adapter import DBAdapter, OBJECT, QUERY, PROJECTION, SEARCH_RESULT
from curate_gpt.store.duckdb_result import DuckDBSearchResult
from curate_gpt.store.metadata import CollectionMetadata
from curate_gpt.utils.vector_algorithms import mmr_diversified_search

logger = logging.getLogger(__name__)

MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384
}

OPENAI_MODEL_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

MODELS = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]


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
        if not self.path:
            self.path = "./duck.db"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.ef_construction = self._validate_ef_construction(self.ef_construction)
        self.ef_search = self._validate_ef_search(self.ef_search)
        self.M = self._validate_m(self.M)
        logger.info(f"Using DuckDB at {self.path}")
        # handling concurrency
        try:
            self.conn = duckdb.connect(self.path, read_only=False)
        except duckdb.IOException as e:
            match = re.search(r'PID (\d+)', str(e))
            if match:
                pid = int(match.group(1))
                logger.info(f"Got {e}.Attempting to kill process with PID: {pid}")
                self.kill_process(pid)
                self.conn = duckdb.connect(self.path, read_only=False)
            else:
                logger.error(f"{e} without PID information.")
                raise
        self.conn.execute("INSTALL vss;")
        self.conn.execute("LOAD vss;")
        self.conn.execute("SET hnsw_enable_experimental_persistence=true;")
        if self.default_model is None:
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
                    "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable")

    def _get_collection_name(self, collection: Optional[str] = None) -> str:
        """
        Get the collection name or the default collection name
        :param collection:
        :return:
        """
        return self._get_collection(collection)

    def _create_table_if_not_exists(self, collection: str, vec_dimension: int, model: str = None):
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

        # metadata row that holds table metadata (CollectionMetadata)
        if model is not None:
            metadata = CollectionMetadata(name=collection, model=model)
        else:
            metadata = CollectionMetadata(name=collection, model=self.default_model)
        metadata_json = json.dumps(metadata.model_dump(exclude_none=True))
        safe_collection_name = f'"{collection}"'
        self.conn.execute(
            f"""
                    INSERT INTO {safe_collection_name} (id, metadata) VALUES ('__metadata__', ?)
                    ON CONFLICT (id) DO NOTHING
                    """, [metadata_json]
        )

    def create_index(self, collection: str, metric: str = "cosine"):
        """
        Create an index for the given collection
        Parameters
        ----------
        collection

        Returns
        -------

        """

        # metrics = ['l2sq', 'cosine', 'ip'] #l2sq,ip
        # for metric in metrics:
        safe_collection_name = f'"{collection}"'
        index_name = f"{collection}_index"
        create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS "{index_name}" ON {safe_collection_name}
            USING HNSW (embeddings) WITH (
                metric='{metric}',
                ef_construction={self.ef_construction},
                ef_search={self.ef_search},
                M={self.M}
            )
        """
        self.conn.execute(create_index_sql)

    def _embedding_function(self, texts: Union[str, List[str]], model: str = None) -> list:
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
            openai_model = model.split(':', 1)[1] if ':' in model else MODELS[1]
            responses = [
                self.openai_client.embeddings.create(input=text, model=openai_model).data[0].embedding
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
        collection = kwargs.get('collection')
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
        collection = kwargs.get('collection')
        ids = [self._id(o, self.id_field) for o in objs]
        existing_ids = set()
        for id_ in ids:
            safe_collection_name = f'"{collection}"'
            result = self.conn.execute(f"SELECT id FROM {safe_collection_name} WHERE id = ?", [id_]).fetchall()
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
            text_field: Union[str, Callable] = None,
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
        self.vec_dimension = self._get_embedding_dimension(model)
        self._create_table_if_not_exists(collection, self.vec_dimension, model=model)
        cm = self.update_collection_metadata(collection, metadata=kwargs)
        if model is None:
            if cm:
                model = cm.model
            if model is None:
                model = self.default_model
        if batch_size is None:
            batch_size = 100000
        if text_field is None:
            text_field = self.text_lookup
        id_field = self.id_field
        num_objs = len(objs) if isinstance(objs, list) else "?"
        cumulative_len = 0
        sql_command = self._generate_sql_command(collection, method)
        sql_command = sql_command.format(collection=collection)
        for next_objs in chunk(objs, batch_size):
            next_objs = list(next_objs)
            docs = [self._text(o, text_field) for o in next_objs]
            docs_len = sum([len(d) for d in docs])
            cumulative_len += docs_len
            if self._is_openai(collection) and cumulative_len > 3000000:
                logger.warning(f"Cumulative length = {cumulative_len}, pausing ...")
                time.sleep(60)
                cumulative_len = 0
            metadatas = [self._dict(o) for o in next_objs]
            ids = [self._id(o, id_field) for o in next_objs]
            embeddings = self._embedding_function(docs, model)
            try:
                self.conn.execute("BEGIN TRANSACTION;")
                self.conn.executemany(sql_command, list(zip(ids, metadatas, embeddings, docs)))
                self.conn.execute("COMMIT;")
            except Exception as e:
                self.conn.execute("ROLLBACK;")
                logger.error(f"Transaction failed: {e}")
                raise
            finally:
                self.create_index(collection)


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

    def search(self, text: str, where: QUERY = None, collection: str = None, limit: int = 10,
               relevance_factor: float = None, **kwargs) -> Iterator[SEARCH_RESULT]:
        """
        Search for objects in the collection that match the given text
        :param text:
        :param where:
        :param collection:
        :param limit:
        :param relevance_factor:
        :param kwargs:
        :return:
        """
        return self._search(text, where, collection, limit, relevance_factor, **kwargs)

    def _search(self, text: str, where: QUERY = None, collection: str = None, limit: int = 10,
                relevance_factor: float = None, model: str = None, **kwargs) -> Iterator[
        SEARCH_RESULT]:
        collection = self._get_collection(collection)
        cm = self.collection_metadata(collection)
        if model is None:
            if cm:
                model = cm.model
            if model is None:
                model = self.default_model
        where_conditions = []
        if where:
            where_conditions.append(where)
        where_clause = " AND ".join(where_conditions)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        if relevance_factor is not None and relevance_factor < 1.0:
            yield from self._diversified_search(text, where, collection, limit, relevance_factor, **kwargs)
            return
        query_embedding = self._embedding_function(text, model)
        print("embedding", query_embedding)
        safe_collection_name = f'"{collection}"'
        results = self.conn.execute(f"""
            SELECT *, (1 - array_cosine_similarity(embeddings::FLOAT[{self.vec_dimension}], 
            {query_embedding}::FLOAT[{self.vec_dimension}])) as distance
            FROM {safe_collection_name}
            {where_clause}
            ORDER BY distance
            LIMIT ?
        """, [limit]).fetchall()
        # first row currently always with distance None as id = '__metadata__'
        results = [r for r in results if r[-1] is not None]
        results = sorted(results, key=lambda x: x[-1])
        yield from self.parse_duckdb_result(results)

    def list_collection_names(self):
        """
        List the names of all collections in the database
        :return:
        """
        result = self.conn.execute("PRAGMA show_tables;").fetchall()
        return [row[0] for row in result]

    def collection_metadata(self, collection_name: Optional[str] = None, include_derived=False, **kwargs
                            ) -> Optional[CollectionMetadata]:
        """
        Get the metadata for the collection
        :param collection_name:
        :param include_derived:
        :param kwargs:
        :return:
        """
        safe_collection_name = f'"{collection_name}"'
        result = self.conn.execute(f"SELECT metadata FROM {safe_collection_name} WHERE id = '__metadata__'").fetchone()
        if result:
            metadata = json.loads(result[0])
            return CollectionMetadata(**metadata)
        return None

    def update_collection_metadata(self, collection: str, **kwargs):
        """
        Update the metadata for the collection
        :param collection:
        :param kwargs:
        :return:
        """
        if collection is None:
            raise ValueError("Collection name must be provided.")

        current_metadata = self.collection_metadata(collection)
        if current_metadata is None:
            current_metadata = CollectionMetadata(**kwargs)
        else:
            current_metadata = current_metadata.model_copy(update=kwargs)
        metadata_json = json.dumps(current_metadata.model_dump(exclude_none=True))
        safe_collection_name = f'"{collection}"'
        self.conn.execute(
            f"""
            UPDATE {safe_collection_name}
            SET metadata = ?
            WHERE id = '__metadata__'
            """, [metadata_json]
        )
        return current_metadata

    def set_collection_metadata(
            self, collection_name: Optional[str], metadata: CollectionMetadata, **kwargs
    ):
        """
        Set the metadata for the collection
        :param collection_name:
        :param metadata:
        :param kwargs:
        :return:
        """
        if collection_name is None:
            raise ValueError("Collection name must be provided.")

        metadata_json = json.dumps(metadata.dict(exclude_none=True))
        safe_collection_name = f'"{collection_name}"'
        self.conn.execute(
            f"""
            UPDATE {safe_collection_name}
            SET metadata = ?
            WHERE id = '__metadata__'
            """, [metadata_json]
        )

    def find(self, where: QUERY = None, projection: PROJECTION = None, collection: str = None,
             include: Optional[List[str]] = None, limit: int = 10, **kwargs) -> Iterator[
        DuckDBSearchResult]:
        """
        Find objects in the collection that match the given query and projection

        :param where: the query to filter the results
        :param projection: TODO: what is this?
        :param collection: name of the collection to search
        :param limit: maximum number of results to return
        :param kwargs:
        :return:

        Parameters
        ----------
        limit
        include
        """
        collection = self._get_collection(collection)
        where_clause = self._parse_where_clause(where) if where else ""
        where_clause = f"WHERE {where_clause}" if where_clause else ""
        safe_collection_name = f'"{collection}"'
        query = f"""
                    SELECT id, metadata, embeddings, documents, NULL as distance
                    FROM {safe_collection_name}
                    {where_clause}
                    LIMIT ?
                """
        results = self.conn.execute(query, [limit]).fetchall()
        yield from self.parse_duckdb_result(results)

    def matches(self, obj: OBJECT, **kwargs) -> Iterator[SEARCH_RESULT]:
        """
        Find objects in the collection that match the given object
        :param obj:
        :param kwargs:
        :return:
        """

        text_field = self.text_lookup
        text = self._text(obj, text_field)
        logger.info(f"Query term: {text}")
        yield from self.search(text, **kwargs)

    def lookup(self, id: str, collection: str = None, **kwargs) -> OBJECT:
        """
        Lookup an object by its id
        :param id: ID of the object to lookup
        :param collection: Name of the collection to search
        :param include: List of fields to include in the output ['metadata', 'embeddings', 'documents']
        :param kwargs:
        :return:
        """
        safe_collection_name = f'"{collection}"'
        result = self.conn.execute(f"""
                SELECT *
                FROM {safe_collection_name}
                WHERE id = ?
            """, [id]).fetchone()
        if isinstance(result, tuple) and len(result) > 1:
            metadata = result[1]
            metadata_object = json.loads(metadata)
            return metadata_object

    def peek(self, collection: str = None, limit=5, **kwargs) -> Iterator[OBJECT]:
        """
        Peek at the first N objects in the collection
        :param collection:
        :param limit:
        :param kwargs:
        :return:
        """
        safe_collection_name = f'"{collection}"'
        results = self.conn.execute(f"""
                SELECT id, metadata, embeddings, documents, NULL as distance
                FROM {safe_collection_name}
                LIMIT ?
            """, [limit]).fetchall()

        yield from self.parse_duckdb_result(results)

    def dump(self, collection: str = None, to_file: Union[str, Path] = None, format: str = "json", **kwargs):
        """
        Dump the collection to a file
        :param collection:
        :param to_file:
        :param format:
        :param kwargs:
        :return:
        """
        if collection is None:
            raise ValueError("Collection name must be provided.")

        collection_name = self._get_collection_name(collection)
        safe_collection_name = f'"{collection_name}"'
        query = f"SELECT id, embeddings, metadata, documents FROM {safe_collection_name}"
        data = self.conn.execute(query).fetchall()
        metadata = self.collection_metadata(collection_name).dict(exclude_none=True)

        result = {
            "metadata": metadata,
            "ids": [row[0] for row in data],
            "embeddings": [row[1] for row in data],
            "metadatas": [json.loads(row[2]) for row in data],
            "documents": [row[3] for row in data]
        }

        if to_file:
            with open(to_file, 'w') as f:
                if format == "json":
                    json.dump(result, f)
                elif format == "yaml":
                    yaml.dump(result, f)
                else:
                    raise ValueError(f"Unsupported format: {format}")

        return result

    def dump_then_load(self, collection: str = None, target: DBAdapter = None,
                       temp_file: Union[str, Path] = "temp_dump.json", format: str = "json"):
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
            raise ValueError("Target must be a DuckDBVSSAdapter instance")
        self.dump(collection=collection, to_file=temp_file, format=format)
        with open(temp_file, 'r') as f:
            if format == "json":
                data = json.load(f)
            elif format == "yaml":
                data = yaml.load(f, Loader=yaml.FullLoader)
            else:
                raise ValueError(f"Unsupported format: {format}")
        metadata = data["metadata"]
        ids = data["ids"]
        embeddings = data["embeddings"]
        metadatas = data["metadatas"]
        documents = data["documents"]
        objects = [
            {"id": id, "embeddings": embedding, "metadata": metadata, "documents": document}
            for id, embedding, metadata, document in zip(ids, embeddings, metadatas, documents)
        ]
        target.remove_collection(collection, exists_ok=True)
        target.set_collection_metadata(collection, metadata)
        batch_size = 5000
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            target.insert(batch, collection=collection)
        Path(temp_file).unlink()

    def _diversified_search(self, text: str, where: QUERY = None, collection: str = None, limit: int = 10,
                            relevance_factor: float = 0.5, **kwargs) -> Iterator[
        SEARCH_RESULT]:
        collection = self._get_collection(collection)
        cm = self.collection_metadata(collection)
        where_conditions = []
        if where:
            for key, value in where.items():
                where_conditions.append(f"json_extract(metadata, '$.{key}') = '{value}'")
        where_clause = " AND ".join(where_conditions)
        if where_clause:
            where_clause = f"WHERE {where_clause}"

        query_embedding = self._embedding_function(text, model=cm.model)
        safe_collection_name = f'"{collection}"'
        results = self.conn.execute(f"""
                    SELECT *, (1 - array_cosine_similarity(embeddings::FLOAT[{self.vec_dimension}], 
                    {query_embedding}::FLOAT[{self.vec_dimension}])) as distance
                    FROM {safe_collection_name}
                    {where_clause}
                    ORDER BY distance 
                    LIMIT ?
                """, [limit * 10]).fetchall()

        # first row currently always with distance None as id = '__metadata__'
        results = [r for r in results if r[-1] is not None]
        results = sorted(results, key=lambda x: x[-1])
        parsed_results = list(self.parse_duckdb_result(results))
        print(parsed_results)
        document_vectors = [np.array(result.embeddings) for result in parsed_results if result.embeddings is not None]
        query_vector = np.array(self._embedding_function(text, model=cm.model))
        if not document_vectors:
            logger.info("The database might be empty. No diversified search results to return.")
            return
        reranked_indices = mmr_diversified_search(
            query_vector=query_vector,
            document_vectors=document_vectors,
            relevance_factor=relevance_factor, top_n=limit
        )
        for idx in reranked_indices:
            yield parsed_results[idx]

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
            if 'model' in metadata and metadata['model'].startswith('openai:'):
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
    def parse_duckdb_result(results) -> Iterator[DuckDBSearchResult]:
        """
        Parse the results from the SQL
        :return: Iterator of DuckDBResult objects
        ----------
        """
        for obj in results:
            yield DuckDBSearchResult(
                id=obj[0],
                metadata=json.loads(obj[1]),
                embeddings=obj[2],
                documents=obj[3],
                distance=obj[4]
            )

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
                        conditions.append(f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) > '{value}'")
                    elif op == "$gte":
                        conditions.append(f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) >= '{value}'")
                    elif op == "$lt":
                        conditions.append(f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) < '{value}'")
                    elif op == "$lte":
                        conditions.append(f"CAST(json_extract_string(metadata, '$.{key}') AS DOUBLE) <= '{value}'")
                    elif op == "$in":
                        conditions.append(
                            f"json_extract_string(metadata, '$.{key}') IN ({', '.join([f'{v}' for v in value])})")
                    elif op == "$not_in":
                        conditions.append(
                            f"json_extract_string(metadata, '$.{key}') NOT IN ({', '.join([f'{v}' for v in value])})")
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
        if model_name is None:
            return MODEL_DIMENSIONS[self.default_model]
        if isinstance(model_name, str):
            if model_name.startswith("openai:"):
                model_key = model_name.split("openai:", 1)[1]
                return OPENAI_MODEL_DIMENSIONS.get(model_key, OPENAI_MODEL_DIMENSIONS["text-embedding-3-small"])
            else:
                if model_name in MODEL_DIMENSIONS:
                    return MODEL_DIMENSIONS[model_name]

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
        if include is None or 'id' in include:
            fields.append("id")
        if include is None or 'metadata' in include:
            fields.append("metadata")
        if include is None or 'embeddings' in include:
            fields.append("embeddings")
        if include is None or 'documents' in include:
            fields.append("documents")
        return ", ".join(fields)
