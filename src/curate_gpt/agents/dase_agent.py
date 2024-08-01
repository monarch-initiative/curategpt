"""Autocomplete objects using RAG."""

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel

from curate_gpt.agents.base_agent import BaseAgent
from curate_gpt.agents.chat_agent import ChatResponse
from curate_gpt.extract import AnnotatedObject
from curate_gpt.store import DBAdapter

logger = logging.getLogger(__name__)

OBJECT = Dict[str, Any]


class PredictedFieldValue(BaseModel):
    id: str
    original_id: Optional[str] = None
    predicted_value: Optional[str] = None
    current_value: Optional[str] = None
    field_predicted: Optional[str] = None


def _dict2str(d: Dict[str, Any]) -> str:
    toks = []
    for k, v in d.items():
        if v:
            toks.append(f"{k}={v}")
    return ", ".join(toks)


@dataclass
class DatabaseAugmentedStructuredExtraction(BaseAgent):
    """
    Extracts structured objects from unstructured documents.

    This implements a standard knowledgebase retrieval augmented generation pattern;
    the knowledge_source is queried for relevant objects; these are presented
    as *examples* to a LLM query, via an extractor.
    """

    document_adapter: DBAdapter = None
    """Adapter to supplementary knowledge in unstructured form."""

    document_adapter_collection: str = None
    """Collection to use for document adapter.
    NOTE: may be deprecated as now collections can be bound to adapters
    """

    default_target_class: ClassVar[str] = "Thing"

    conversation: List[Dict[str, Any]] = None  # TODO
    conversation_mode: bool = False  # TODO
    relevance_factor: float = 0.5
    """Relevance factor for diversifying search results using MMR."""

    max_background_document_size: int = 1000
    """TODO: more sophisticated way to estimate size of background document."""

    background_document_limit: int = 3
    """Number of background documents to use. TODO: more sophisticated way to estimate."""

    default_masked_fields: List[str] = field(default_factory=lambda: ["original_id"])

    def extract(
        self,
        text: Union[str, ChatResponse],
        target_class: str = None,
        feature_fields: List[str] = None,
        generate_background=False,
        collection: str = None,
        rules: List[str] = None,
        fields_to_mask: List[str] = None,
        fields_to_predict: List[str] = None,
        merge=True,
        **kwargs,
    ) -> AnnotatedObject:
        """
        Populate structured object from text

        :param seed:
        :param target_class:
        :param context_property:
        :param generate_background:
        :param collection:
        :param rules:
        :param kwargs:
        :return:
        """
        if fields_to_mask is None:
            fields_to_mask = []
        extractor = self.extractor
        if not target_class:
            cm = self.knowledge_source.collection_metadata(collection)
            if not cm:
                raise ValueError(f"Invalid collection: {collection}")
            target_class = cm.object_type
        if not target_class:
            target_class = self.default_target_class

        annotated_examples = []
        logger.debug(f"Searching for seed: {text}")
        for obj, _, _obj_meta in self.knowledge_source.search(
            text,
            relevance_factor=self.relevance_factor,
            collection=collection,
            **kwargs,
        ):
            min_obj = {
                k: v
                for k, v in obj.items()
                if v
                and isinstance(v, str)
                and k not in fields_to_mask
                and (not fields_to_predict or k in fields_to_predict)
            }
            if not min_obj:
                continue
            ae = AnnotatedObject(object=obj)
            annotated_examples.append(ae)
        if not annotated_examples:
            raise ValueError(f"No examples found for seed: {text}")
        docs = []
        if self.document_adapter:
            logger.debug("Adding background knowledge.")
            for _obj, _, obj_meta in self.document_adapter.search(
                text,
                limit=self.background_document_limit,
                collection=self.document_adapter_collection,
            ):
                obj_text = obj_meta["document"]
                # TODO: use tiktoken to estimate
                obj_text = obj_text[0 : self.max_background_document_size]
                docs.append(obj_text)
        if generate_background:
            # prompt = f"Generate a comprehensive description about the {target_class} with {context_property} = {seed}"
            response = extractor.model.prompt(
                f"Describe the {target_class} in the following text: {text}"
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
        ao = extractor.extract(
            f"Generate a {target_class} from the following text: {text}",
            target_class=target_class,
            examples=annotated_examples,
            background_text=background,
            rules=rules,
            min_examples=2,
        )
        return ao
