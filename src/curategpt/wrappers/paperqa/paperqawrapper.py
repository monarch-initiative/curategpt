from curategpt.wrappers import BaseWrapper
import os
import logging
import tempfile
import asyncio
import shutil
from pathlib import Path
import importlib.util
import pkg_resources
import sys

logger = logging.getLogger(__name__)

# Check paperqa version
try:
    paperqa_version = pkg_resources.get_distribution("paperqa").version
    logger.info(f"Using PaperQA version: {paperqa_version}")
except Exception:
    paperqa_version = "unknown"
    logger.warning("Unable to determine PaperQA version")

# Import paperqa components with version handling
try:
    from paperqa import Docs
    # In newer versions, VectorStore implementations might be in different locations
    try:
        from paperqa import NumpyVectorStore, QdrantVectorStore
    except ImportError:
        # Try alternative import locations for older versions
        try:
            from paperqa.vectorstores import NumpyVectorStore, QdrantVectorStore
        except ImportError:
            # Define minimal versions if needed for compatibility
            class NumpyVectorStore:
                pass
            
            class QdrantVectorStore:
                pass
            
            logger.warning("Using minimal VectorStore implementations")
except ImportError:
    logger.error("Failed to import paperqa. Please install it with: pip install paperqa")
    raise


class PaperQAWrapper(BaseWrapper):
    """PaperQA wrapper with flexible storage options for persistence."""
    
    name = "paperqa"

    def __init__(self, collection_name="paperqa_docs", db_path=None, use_qdrant=False, **kwargs):
        super().__init__(**kwargs)
        self.collection_name = collection_name

        try:
            # Determine storage directory path
            if db_path:
                self.db_path = Path(db_path)
            else:
                self.db_path = Path("./paperqa_db")
            
            # Create the directory if it doesn't exist
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Use NumpyVectorStore for maximum compatibility
            self.vector_store = NumpyVectorStore()
            self.docs = Docs(texts_index=self.vector_store, name=collection_name)
            
            # Set up pickle persistence path
            self.save_path = self.db_path / f"{collection_name}.pkl"
            self.persistence_type = "pickle"
            
            # Try to load existing data if available
            if self.save_path.exists():
                self._load_docs()
                logger.info(f"Loaded existing documents from {self.save_path}")
            else:
                logger.info(f"Creating new collection: {collection_name}")
                
        except ImportError as e:
            logger.error(f"Error initializing PaperQA: {e}")
            raise

    def _save_docs(self):
        """Save the docs to disk using pickle."""
        import pickle
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to a temporary file first to avoid corruption
        tmp_path = f"{self.save_path}.tmp"
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(self.docs, f)
            # Atomically replace the old file with the new one
            shutil.move(tmp_path, self.save_path)
            logger.info(f"Saved documents to {self.save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving documents: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            return False

    def _load_docs(self):
        """Load the docs from disk using pickle."""
        if not self.save_path.exists():
            return False
            
        import pickle
        try:
            with open(self.save_path, 'rb') as f:
                self.docs = pickle.load(f)
            logger.info(f"Successfully loaded documents from {self.save_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return False

    def search(self, query, limit=10, **kwargs):
        """Search using PaperQA and return in CurateGPT format."""
        try:
            # Run async query in a synchronous context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                answer = loop.run_until_complete(self.docs.aquery(query))
            finally:
                loop.close()
            
            # Return results in the expected format
            for i, context in enumerate(answer.contexts[:limit]):
                yield (
                    {
                        "text": context.context,
                        "source": context.text.doc.citation if hasattr(context.text.doc, 'citation') else f"Document: {context.text.doc.docname}",
                        "paper_name": context.text.doc.docname,
                    },
                    context.score,
                    {"score": context.score}
                )
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    async def aadd(self, file_path, **kwargs):
        """Add documents to PaperQA with persistence."""
        try:
            # Verify file exists and is readable
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check if it's a valid PDF
            if path.suffix.lower() == '.pdf':
                try:
                    # Try to verify PDF is valid
                    try:
                        import pymupdf
                        _ = pymupdf.open(str(path))
                    except ImportError:
                        # Try PyMuPDF under its fitz name
                        import fitz
                        _ = fitz.open(str(path))
                except ImportError:
                    logger.warning("PyMuPDF not available - can't validate PDF")
                except Exception as e:
                    raise ValueError(f"Invalid PDF file: {e}")

            # Get the docname if not provided in kwargs
            if 'docname' not in kwargs:
                kwargs['docname'] = path.stem

            # Add the document
            result = await self.docs.aadd(str(path), **kwargs)
            
            # Save if using pickle persistence
            self._save_docs()
            
            return result

        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            raise

