import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import tempfile
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

    def upload_collection(self, objects, metadata, repo_id, private=False, **kwargs):
        """
        Upload an entire collection to a Hugging Face repository.

        :param objects: The objects to upload.
        :param metadata: The metadata associated with the collection.
        :param repo_id: The repository ID on Hugging Face.
        :param private: Whether the repository should be private.
        :param kwargs: Additional arguments such as batch size or metadata options.
        """
        # Transform metadata into VenomX format using the private method
        venomx_metadata = self._transform_metadata_to_venomx(metadata)

        # Define the file names to be saved in the current working directory
        embedding_file = "embeddings.parquet"
        metadata_file = "metadata.yaml"

        # Save objects and metadata to files in the cwd
        pd.DataFrame(objects).to_parquet(embedding_file)  # Serialize objects to Parquet
        with open(metadata_file, "w") as f:
            yaml.dump(venomx_metadata, f, sort_keys=False)

        # Ensure the repository exists
        self._create_repo(repo_id, private=private)

        # Upload files to the repository
        self._upload_files(repo_id, {
            embedding_file: embedding_file,  # Directly reference the file names
            metadata_file: metadata_file
        })

    def _transform_metadata_to_venomx(self, metadata):
        """
        Transform metadata from ChromaDB format to VenomX format.

        :param metadata: Metadata object from store
        :return: A dictionary formatted according to VenomX
        """

        prefixes = metadata.prefixes if hasattr(metadata,
                                                'prefixes') and metadata.prefixes else {}

        venomx_metadata = {
            "description": metadata.description or "No description provided",
            "prefixes": prefixes,
            "model": {
                "name": metadata.model or "unknown",
                # Default to a known model if not specified
            },
            "model_input_method": {
                "description": "Simple pass through of labels only",
                "fields": ["rdfs:label"]
                # Adjust fields based on actual data structure if needed
            },
            "dataset": {
                "name": metadata.name or "Unknown Dataset",
                "url": metadata.source or "Unknown URL"
                # Adjust based on available metadata
            }
        }

        # Enrich VenomX format with annotations if available
        if metadata.annotations:
            venomx_metadata["annotations"] = metadata.annotations

        # Include any additional fields from metadata that are relevant
        if hasattr(metadata, 'extra_fields') and metadata.extra_fields:
            venomx_metadata.update(metadata.extra_fields)

        return venomx_metadata

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
