import logging
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from huggingface_hub import HfApi, create_repo, get_token

logger = logging.getLogger(__name__)
HF_DOWNLOAD_PATH = Path(__file__).resolve().parents[4]
HF_DOWNLOAD_PATH = HF_DOWNLOAD_PATH / "hf_download"

@dataclass
class HuggingFaceAgent:

    api: HfApi = None

    def __post_init__(self):
        self.api = HfApi()
        self.token = get_token()
        # set export HUGGING_FACE_HUB_TOKEN="your_access_token"

    def upload(self, objects, metadata, repo_id, private=False, **kwargs):
        """
        Upload an entire collection to a Hugging Face repository.

        :param objects: The objects to upload.
        :param metadata: The metadata associated with the collection.
        :param repo_id: The repository ID on Hugging Face.
        :param private: Whether the repository should be private.
        :param kwargs: Additional arguments such as batch size or metadata options.
        """

        embedding_file = "embeddings.parquet"
        metadata_file = "metadata.yaml"
        print("\n\n")
        print(objects[0][0])
        print(objects[0][2]['_embeddings'])
        print(objects[0][2]['documents'])
        print("\n\n")
        try:
            df = pd.DataFrame(data=[(obj[0], obj[2]['_embeddings'], obj[2]['document']) for obj in objects])
        except Exception as e:
            # df = pd.DataFrame(data=[(obj[0], obj[2]['_embeddings'], obj[2]['documents']) for obj in objects])
            # logger.info(f"df changed")
            raise ValueError(f"Creation of Dataframe not successful: {e}") from e


        with ExitStack() as stack:
            tmp_parquet = stack.enter_context(tempfile.NamedTemporaryFile(suffix=".parquet", delete=True))
            tmp_yaml = stack.enter_context(tempfile.NamedTemporaryFile(suffix=".yaml", delete=True))

            embedding_path = tmp_parquet.name
            metadata_path = tmp_yaml.name

            df.to_parquet(path=embedding_path, index=False)
            with open(metadata_path, "w") as f:
                yaml.dump(metadata.model_dump(), f)

            self._create_repo(repo_id, private=private)

            self._upload_files(repo_id, {
                embedding_path : repo_id + "/" + embedding_file,
                metadata_path : repo_id + "/" + metadata_file
            })

    def upload_duckdb(self, objects, metadata, repo_id, private=False, **kwargs):
        """
        Upload an entire collection to a Hugging Face repository.

        :param objects: The objects to upload.
        :param metadata: The metadata associated with the collection.
        :param repo_id: The repository ID on Hugging Face.
        :param private: Whether the repository should be private.
        :param kwargs: Additional arguments such as batch size or metadata options.
        """

        embedding_file = "embeddings.parquet"
        metadata_file = "metadata.yaml"
        try:
            df = pd.DataFrame(data=[(obj[0], obj[2]['_embeddings'], obj[2]['documents']) for obj in objects])
        except Exception as e:
            raise ValueError(f"Creation of Dataframe not successful: {e}") from e

        with ExitStack() as stack:
            tmp_parquet = stack.enter_context(tempfile.NamedTemporaryFile(suffix=".parquet", delete=True))
            tmp_yaml = stack.enter_context(tempfile.NamedTemporaryFile(suffix=".yaml", delete=True))

            embedding_path = tmp_parquet.name
            metadata_path = tmp_yaml.name

            df.to_parquet(path=embedding_path, index=False)
            with open(metadata_path, "w") as f:
                yaml.dump(metadata.model_dump(), f)

            self._create_repo(repo_id, private=private)

            self._upload_files(repo_id, {
                embedding_path : repo_id + "/" + embedding_file,
                metadata_path : repo_id + "/" + metadata_file
            })

    def _create_repo(self, repo_id: str, private: bool = False):
        """
        Create a new repository on Hugging Face Hub.

        :param repo_id: The repository ID, e.g., "biomedical-translator/[your repo name]".
        :param private: Whether the repository is private.
        """
        try:
            create_repo(repo_id=repo_id, token=self.token, repo_type="dataset", private=private)
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
        try:
            for local_path, repo_path in files.items():
                self.api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                logger.info(f"Uploaded {local_path} to {repo_path} in {repo_id}")
        except Exception as e:
            logger.error(f"Failed to upload files to {repo_id} on Hugging Face: {e}")
            raise

    def cached_download(
        self,
        repo_id: str,
        repo_type: str,
        filename: str
    ):
        download_path = self.api.hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=filename,
            token=self.token,
        )

        return download_path




