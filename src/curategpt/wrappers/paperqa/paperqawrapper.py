import asyncio
import logging
import os
from dataclasses import dataclass

from paperqa import Settings
from paperqa.agents.main import agent_query
from paperqa.agents.search import get_directory_index

from curategpt.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)


@dataclass
class PaperQAWrapper(BaseWrapper):
    """
    A wrapper for PaperQA to search Alzheimer's papers.

    This wrapper uses PaperQA to search through a corpus of research papers.
    It assumes papers have already been indexed using PaperQA's CLI tools.
    """

    name = "paperqa"

    def __post_init__(self) -> None:
        pqa_home = os.environ.get("PQA_HOME")
        if not pqa_home:
            raise ValueError("PQA_HOME environment variable is not set!")
        self.settings = Settings(paper_directory=pqa_home)
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        async def _check_and_build_index():
            try:
                # given we build with cli this should work
                await get_directory_index(settings=self.settings, build=False)
                print("Existing index found")
                return True
            except Exception as e:
                if "was empty" in str(e):
                    print("Index is empty, building now...")
                    try:
                        # Build the index
                        await get_directory_index(settings=self.settings, build=True)
                        print("Index built successfully")
                        return True
                    except Exception as build_err:
                        print(f"Error building index: {build_err}")
                        return False
                else:
                    print(f"Error accessing index: {e}")
                    return False

        asyncio.run(_check_and_build_index())

    def search(self, query, limit=10, **kwargs):
        """Search for documents matching the query using PaperQA."""
        logger.info(f"Searching for: {query}")

        async def _search():
            try:
                response = await agent_query(
                    query=query,
                    settings=self.settings
                )
                return response

            except Exception as e:
                logger.error(f"Error with agent_query: {e}")
                raise e

        return asyncio.run(_search())
