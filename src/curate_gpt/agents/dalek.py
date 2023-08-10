"""Retrieval Augmented Generation (RAG) Base Class."""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, ClassVar

import yaml

from curate_gpt.extract import AnnotatedObject, Extractor
from curate_gpt.store import DBAdapter

logger = logging.getLogger(__name__)

OBJECT = Dict[str, Any]


@dataclass
class DatabaseAugmentedExtractor:
    """
    An agent to extract knowledge with augmentation from databases.
    """

    kb_adapter: DBAdapter = None
    """Adapter to structured knowledge base"""

    document_adapter: DBAdapter = None
    """Adapter to supplementary knowledge in unstructured form."""

    document_adapter_collection: str = None

    extractor: Extractor = None
    """Engine for extracting structured records from in-context prompts"""

    default_target_class: ClassVar[str] = "Thing"

    conversation: List[Dict[str, Any]] = None # TODO
    conversation_mode: bool = False # TODO
    relevance_factor: float = 0.5
    """Relevance factor for diversifying search results using MMR."""

    max_background_document_size: int = 1000
    """TODO: more sophisticated way to estimate size of background document."""

    background_document_limit: int = 3
    """Number of background documents to use. TODO: more sophisticated way to estimate."""

    def generate_extract(
        self,
        text: str,
        target_class: str = None,
        context_property: str = None,
        generate_background=False,
        collection: str = None,
        rules: List[str] = None,
        **kwargs,
    ) -> AnnotatedObject:
        """
        Extract structured object using text seed and background knowledge.

        :param text:
        :param kwargs:
        :return:
        """
        extractor = self.extractor
        if not target_class:
            target_class = self.kb_adapter.collection_metadata(collection).object_type
        if not target_class:
            target_class = self.default_target_class
        if context_property is None:
            context_property = "label"
        gen_prompt = f"Structured representation of entity with {context_property} = " + "{text}"
        annotated_examples = []
        for obj, _, obj_meta in self.kb_adapter.search(
            text, relevance_factor=self.relevance_factor, collection=collection, **kwargs
        ):
            if not context_property:
                logger.warning(f"We recommend specifying a context property for {target_class}")
                obj_text = json.dumps({k: v for k, v in obj.items() if v}, sort_keys=False)
            else:
                if context_property not in obj:
                    logger.debug(f"Skipping object because it does not have {context_property}")
                    continue
                obj_text = obj[context_property]
            # obj_text = obj_meta["document"]
            ae = AnnotatedObject(object=obj, annotations={"text": gen_prompt.format(text=obj_text)})
            annotated_examples.append(ae)
        docs = []
        if self.document_adapter:
            for obj, _, obj_meta in self.document_adapter.search(
                text, limit=self.background_document_limit, collection=self.document_adapter_collection
            ):
                obj_text = obj_meta["document"]
                # TODO: use tiktoken to estimate
                obj_text = obj_text[0:self.max_background_document_size]
                docs.append(obj_text)
        gen_text = gen_prompt.format(text=text)
        if generate_background:
            response = extractor.model.prompt(
                f"Generate a comprehensive description about the {target_class} with {context_property} = {text}"
            )
            if docs is None:
                docs = []
            docs.append(response.text())
        if docs:
            # TODO: experiment with placing after examples.
            background = "BACKGROUND:"
            background += "\n\n" + "\n\n".join(docs)
            background += "\n---\n"
            logger.info(f"Background: {background}")
        else:
            background = None
        return extractor.extract(
            gen_text,
            target_class=target_class,
            examples=annotated_examples,
            background_text=background,
            rules=rules,
        )
