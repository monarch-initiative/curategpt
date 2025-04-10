import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from paperqa import Docs, Settings
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
      print(f"Searching for: {query}")
      async def _search():
          try:
              response = await agent_query(
                  query=query,
                  settings=self.settings
              )

              print(f"Query response contains {len(response.session.contexts)} contexts")

              results = []
              for idx, context in enumerate(response.session.contexts[:limit]):
                  # Create a result object that matches the standard format
                  obj = {
                      "id": f"paper_{idx}",
                      "title": context.text.doc.docname,
                      "abstract": context.context,
                      "citation": (getattr(context.text.doc, "citation", None) or
                               f"Document: {context.text.doc.docname}"),
                  }

                  # Calculate a pseudo-distance (lower is better, use rank position)
                  # Normalize to be between 0-1 like other distance metrics
                  distance = idx / (len(response.session.contexts) or 1)

                  # Add metadata for consistency with other wrappers
                  metadata = {
                      "document": obj["abstract"],
                      "paper_name": context.text.doc.docname
                  }

                  # Append as a tuple of (obj, distance, metadata)
                  results.append((obj, distance, metadata))

              return iter(results)
          except Exception as e:
              self._fallback_search(query, limit)
              print(f"Error with agent_query: {e}")
              raise e

      return asyncio.run(_search())

  def _fallback_search(self, query, limit=10):
      """Fallback search method when agent_query fails."""
      try:
          async def _direct_docs_search():
              docs = Docs()
              pdf_path = Path(self.settings.paper_directory)
              pdf_files = list(pdf_path.glob("*.pdf"))

              for pdf_file in pdf_files:
                  await docs.aadd(str(pdf_file))

              answer = await docs.aquery(query, settings=self.settings)

              results = []
              for idx, context in enumerate(answer.contexts[:limit]):
                  # Create a result object that matches the standard format
                  obj = {
                      "id": f"paper_{idx}",
                      "title": context.text.doc.docname,
                      "abstract": context.context,
                      "citation": (getattr(context.text.doc, "citation", None) or
                              f"Document: {context.text.doc.docname}"),
                  }

                  # Calculate a pseudo-distance (lower is better, use rank position)
                  # Normalize to be between 0-1 like other distance metrics
                  distance = idx / (len(answer.contexts) or 1)

                  # Add metadata for consistency with other wrappers
                  metadata = {
                      "document": obj["abstract"],
                      "paper_name": context.text.doc.docname
                  }

                  # Append as a tuple of (obj, distance, metadata)
                  results.append((obj, distance, metadata))

              return iter(results)

          return asyncio.run(_direct_docs_search())
      except Exception as e:
          print(f"Error in fallback search: {e}")
          return iter([])
