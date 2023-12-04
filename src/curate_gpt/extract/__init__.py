"""
CurateGPT Extractors.

These handle connections to (remote or local) LLMs,
and can also extract structured objects from text.

* Base class: :class:`Extractor`
"""

from .basic_extractor import BasicExtractor
from .extractor import AnnotatedObject, Extractor
from .openai_extractor import OpenAIExtractor
from .recursive_extractor import RecursiveExtractor

__all__ = [
    "BasicExtractor",
    "AnnotatedObject",
    "Extractor",
    "RecursiveExtractor",
    "OpenAIExtractor",
]
