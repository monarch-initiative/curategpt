"""Wrapper for the PaperQA library to search Alzheimer's papers."""

import logging
import os
import asyncio
from typing import Any, Dict, List, Optional, Iterator

from paperqa import Settings
from paperqa.agents.main import agent_query

from curategpt.wrappers.base_wrapper import BaseWrapper

logger = logging.getLogger(__name__)


class PaperQAWrapper(BaseWrapper):
    """
    A wrapper for PaperQA to search Alzheimer's papers.
    
    This wrapper uses PaperQA to search through a corpus of research papers.
    It assumes papers have already been indexed using PaperQA's CLI tools
    and can be found via the PQA_HOME environment variable.
    """

    name = "paperqa"
    
    def __init__(self, **kwargs):
        """
        Initialize the PaperQA wrapper.
        
        Uses the PQA_HOME environment variable to find indexed papers.
        """
        super().__init__(**kwargs)
        
        # Use default PaperQA settings 
        # This will look for papers in PQA_HOME environment variable
        self.settings = Settings()
        
        logger.info("Initialized PaperQA wrapper for Alzheimer's papers")

    def external_search(
        self,
        text: str,
        expand: bool = False,
        limit: Optional[int] = 10,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Search Alzheimer's papers using PaperQA.
        
        Args:
            text: Query string
            expand: Whether to expand the query (not used)
            limit: Maximum number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List of dictionaries with paper information
        """
        try:
            # Run the async query in a synchronous context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Perform the query
                response = loop.run_until_complete(
                    agent_query(
                        query=text, 
                        settings=self.settings, 
                        answer_with_sources=True
                    )
                )
            finally:
                loop.close()
            
            # Format results as list of dicts
            results = []
            
            # Add the source contexts
            for i, context in enumerate(response.contexts[:limit]):
                result = {
                    "id": f"paperqa_{i}",
                    "text": context.text,
                    "title": context.source_name if hasattr(context, 'source_name') else f"Source {i+1}",
                    "citation": context.citation if hasattr(context, 'citation') else "",
                    "score": context.score if hasattr(context, 'score') else 0.0,
                }
                results.append(result)
                
            return results
        except Exception as e:
            logger.error(f"Error searching with PaperQA: {e}")
            return []

    def objects_by_ids(self, object_ids: List[str]) -> List[Dict]:
        """Not implemented for PaperQA wrapper."""
        logger.warning("PaperQA doesn't support direct ID lookup")
        return []