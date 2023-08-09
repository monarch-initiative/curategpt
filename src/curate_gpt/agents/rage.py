"""Retrieval Augmented Generation (RAG) Base Class."""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from curate_gpt.extract import Extractor, AnnotatedObject
from curate_gpt.store import DBAdapter

logger = logging.getLogger(__name__)

OBJECT = Dict[str, Any]


@dataclass
class RetrievalAugmentedExtractor:
    """
    TODO: rename:

    AUGER
    SPICE
    KRAGE
    """
    kb_adapter: DBAdapter = None
    """Adapter to structured knowledge base"""

    document_adapter: DBAdapter = None
    """Adapter to supplementary knowledge in unstructured form."""

    document_adapter_collection: str = None

    extractor: Extractor = None
    """Engine for extracting structured records from in-context prompts"""

    conversation: List[Dict[str, Any]] = None
    conversation_mode: bool = False
    relevance_factor: float = 0.5

    def generate_extract(self, text: str, target_class, context_property: str = None, **kwargs) -> AnnotatedObject:
        """
        Perform Retrieval Augmented Generation Extraction (RAGE) from text.

        :param text:
        :param kwargs:
        :return:
        """
        extractor = self.extractor
        if context_property is None:
            context_property = "label"
        gen_prompt = f"Structured representation of entity with {context_property} = " + "{text}"
        print(f"KWARGS={kwargs}")
        annotated_examples = []
        for obj, _, obj_meta in self.kb_adapter.search(text, relevance_factor=self.relevance_factor, **kwargs):
            if context_property not in obj:
                continue
            obj_text = obj[context_property]
            # obj_text = obj_meta["document"]
            ae = AnnotatedObject(object=obj, annotations={"text": gen_prompt.format(text=obj_text)})
            annotated_examples.append(ae)
        docs = []
        if self.document_adapter:
            for obj, _, obj_meta in self.document_adapter.search(text, limit=3, collection=self.document_adapter_collection):
                obj_text = obj_meta["document"]
                # TODO: customize
                obj_text = obj_text[0:400]
                docs.append(obj_text)
        gen_text = gen_prompt.format(text=text)
        if docs:
            # TODO: experiment with placing after examples.
            background = "BACKGROUND:"
            background += "\n\n" + "\n\n".join(docs)
            background += "\n---\n"
        else:
            background = None
        return extractor.extract(gen_text, target_class=target_class, examples=annotated_examples, background_text=background)
