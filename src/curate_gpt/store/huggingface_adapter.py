import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import yaml
from huggingface_hub import HfApi, create_repo

from curate_gpt import DBAdapter

logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceAdapter(DBAdapter):
    """
    Adapter class for interacting with Hugging Face Hub.
    Implements only the necessary functionality to upload collections, with dummy methods for other operations.
    """

    def insert(self, objs, collection: str = None, **kwargs):
        """
        Dummy insert method. Not implemented for HuggingFaceAdapter.
        """
        pass

    def list_collection_names(self) -> List[str]:
        """
        Dummy list_collection_names method. Not implemented for HuggingFaceAdapter.
        """
        return []

    def collection_metadata(self, collection_name: Optional[str] = None, include_derived=False, **kwargs):
        """
        Dummy collection_metadata method. Not implemented for HuggingFaceAdapter.
        """
        return None

    def lookup(self, id: str, collection: str = None, **kwargs):
        """
        Dummy lookup method. Not implemented for HuggingFaceAdapter.
        """
        pass

    def matches(self, obj, **kwargs):
        """
        Dummy matches method. Not implemented for HuggingFaceAdapter.
        """
        pass

    def peek(self, collection: str = None, limit=5, **kwargs):
        """
        Dummy peek method. Not implemented for HuggingFaceAdapter.
        """
        return []

    def search(self, text: str, where: dict = None, collection: str = None, **kwargs):
        """
        Dummy search method. Not implemented for HuggingFaceAdapter.
        """
        return []

    def upload_collection(self, collection: str, repo_id: str, private: bool = False, **kwargs):
        """
        Upload an entire collection to a Hugging Face repository.

        :param collection: The name of the collection to upload.
        :param repo_id: The repository ID on Hugging Face.
        :param private: Whether the repository should be private.
        :param kwargs: Additional arguments such as batch size or metadata options.
        """
        # Ensure the repository exists
        self._create_repo(repo_id, private=private)

        # Fetch objects and metadata from the collection
        objects = list(self.find(collection=collection))  # <- fix

        metadata = self.collection_metadata(collection)

        # Save objects and metadata to temporary files
        embedding_file = f"{collection}_embeddings.parquet"
        metadata_file = f"{collection}_metadata.yaml"

        pd.DataFrame(objects).to_parquet(embedding_file)  # Example serialization to Parquet
        with open(metadata_file, "w") as f:
            yaml.dump(metadata.model_dump(), f, sort_keys=False)

        # Upload files to the repository
        self._upload_files(repo_id, {embedding_file: embedding_file, metadata_file: metadata_file})

    def _create_repo(self, repo_id: str, private: bool = False):
        """
        Create a new repository on Hugging Face Hub.

        :param repo_id: The repository ID, e.g., "biomedical-translator/[your repo name]".
        :param private: Whether the repository is private.
        """
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", private=private)
            logger.info(f"Repository {repo_id} created successfully on Hugging Face.")
        except Exception as e:
            logger.error(f"Failed to create repository {repo_id} on Hugging Face: {e}")
            raise

    def _upload_files(self, repo_id: str, files: Dict[str, str]):
        """
        Upload files to a Hugging Face repository.

        :param repo_id: The repository ID on Hugging Face.
        :param files: A dictionary with local file paths as keys and paths in the repository as values.
        """
        api = HfApi()
        try:
            for local_path, repo_path in files.items():
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                logger.info(f"Uploaded {local_path} to {repo_path} in {repo_id}")
        except Exception as e:
            logger.error(f"Failed to upload files to {repo_id} on Hugging Face: {e}")
            raise
