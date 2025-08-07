import asyncio
import logging
import os
from dataclasses import dataclass

from paperqa import Settings
from paperqa.agents.main import agent_query
from paperqa.agents.search import get_directory_index
from paperqa.settings import IndexSettings

from curategpt.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)


@dataclass
class PaperQAWrapper(BaseWrapper):
    """
    A wrapper for PaperQA to search through corpus of research papers.

    This wrapper uses PaperQA to search through a corpus of research papers.
    It assumes papers have already been indexed using PaperQA's CLI tools.
    """

    name = "paperqa"
    corpus_id: str = ""  # Optional corpus identifier ("" for default, "2" for second corpus)

    def __post_init__(self) -> None:
        # Use corpus-specific environment variables
        pqa_home_var = f"PQA_HOME{self.corpus_id}" if self.corpus_id else "PQA_HOME"
        pqa_index_var = f"PQA_INDEX{self.corpus_id}" if self.corpus_id else "PQA_INDEX"
        
        pqa_home = os.environ.get(pqa_home_var)
        if not pqa_home:
            raise ValueError(f"{pqa_home_var} environment variable is not set!")
        
        self.settings = Settings(paper_directory=pqa_home)
        
        # Allow optional specification of existing index
        pqa_index = os.environ.get(pqa_index_var)
        if pqa_index:
            self.settings.agent.index.name = pqa_index
            # Set the index directory to be in the paper directory
            self.settings.agent.index.index_directory = f"{pqa_home}/.pqa/indexes"
            logger.info(f"Using specified index: {pqa_index} for corpus {self.corpus_id or 'default'}")
            logger.info(f"Index directory: {self.settings.agent.index.index_directory}")
        
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
