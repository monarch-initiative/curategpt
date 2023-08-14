"""Retrieval Augmented Generation (RAG) Base Class."""
import json
import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Union

import yaml

from curate_gpt.extract import AnnotatedObject, Extractor
from curate_gpt.store import DBAdapter

logger = logging.getLogger(__name__)

OBJECT = Dict[str, Any]


def _dict2str(d: Dict[str, Any]) -> str:
    toks = []
    for k, v in d.items():
        if v:
            toks.append(f"{k}={v}")
    return ", ".join(toks)


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

    conversation: List[Dict[str, Any]] = None  # TODO
    conversation_mode: bool = False  # TODO
    relevance_factor: float = 0.5
    """Relevance factor for diversifying search results using MMR."""

    max_background_document_size: int = 1000
    """TODO: more sophisticated way to estimate size of background document."""

    background_document_limit: int = 3
    """Number of background documents to use. TODO: more sophisticated way to estimate."""

    def generate_extract(
        self,
        seed: Union[str, Dict[str, Any]],
        target_class: str = None,
        context_property: str = None,
        generate_background=False,
        collection: str = None,
        rules: List[str] = None,
        **kwargs,
    ) -> AnnotatedObject:
        """
        Extract structured object using text seed and background knowledge.

        :param seed:
        :param target_class:
        :param context_property:
        :param generate_background:
        :param collection:
        :param rules:
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
        if isinstance(seed, str):
            seed = {context_property: seed}
        if isinstance(seed, dict):
            context_properties = list(seed.keys())
        else:
            if context_property:
                context_properties = [context_property]
            else:
                context_properties = []

        def gen_prompt_f(obj: Union[str, Dict], prefix="Structured representation of") -> str:
            if isinstance(obj, dict):
                if not context_properties:
                    min_obj = {k: v for k, v in obj.items() if v and isinstance(v, str)}
                else:
                    min_obj = {k: obj[k] for k in context_properties if k in obj}
                if min_obj:
                    as_text = yaml.safe_dump(min_obj, sort_keys=True).strip()
                    return f"{prefix} {target_class} with {as_text}"
                else:
                    return f"{prefix} {target_class}:"
            elif isinstance(obj, str):
                return f"{prefix} {target_class} with {context_property} = {obj}"
            else:
                raise ValueError(f"Invalid type for obj: {type(obj)} //  {obj}")

        annotated_examples = []
        seed_search_term = seed if isinstance(seed, str) else yaml.safe_dump(seed, sort_keys=True)
        for obj, _, obj_meta in self.kb_adapter.search(
            seed_search_term,
            relevance_factor=self.relevance_factor,
            collection=collection,
            **kwargs,
        ):
            ae = AnnotatedObject(object=obj, annotations={"text": gen_prompt_f(obj)})
            annotated_examples.append(ae)
        docs = []
        if self.document_adapter:
            for obj, _, obj_meta in self.document_adapter.search(
                seed_search_term,
                limit=self.background_document_limit,
                collection=self.document_adapter_collection,
            ):
                obj_text = obj_meta["document"]
                # TODO: use tiktoken to estimate
                obj_text = obj_text[0 : self.max_background_document_size]
                docs.append(obj_text)
        gen_text = gen_prompt_f(seed)
        if generate_background:
            # prompt = f"Generate a comprehensive description about the {target_class} with {context_property} = {seed}"
            prompt = gen_prompt_f(seed, prefix="Generate a comprehensive description about the")
            response = extractor.model.prompt(prompt)
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
