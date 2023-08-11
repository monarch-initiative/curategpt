"""Chat with a KB."""
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, ClassVar, Optional

import yaml
from pydantic import BaseModel

from curate_gpt.extract import AnnotatedObject, Extractor
from curate_gpt.store import DBAdapter
from llm import Conversation

logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    """Response from chat engine."""

    response: str
    """Text of response."""

    prompt: str
    """Prompt used to generate response."""

    formatted_response: str = None

    references: Optional[Dict[str, Any]] = None


def replace_references_with_links(text):
    pattern = r'\[(\d+)\]'
    replacement = lambda m: f"[{m.group(1)}](#ref-{m.group(1)})"
    return re.sub(pattern, replacement, text)


@dataclass
class ChatEngine:
    """
    An agent to extract knowledge with augmentation from databases.
    """

    kb_adapter: DBAdapter = None
    """Adapter to structured knowledge base"""

    extractor: Extractor = None

    relevance_factor: float = 0.5
    """Relevance factor for diversifying search results using MMR."""

    conversation_id: Optional[str] = None

    def chat(
        self,
        query: str,
        conversation: Optional[Conversation] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Extract structured object using text seed and background knowledge.

        :param text:
        :param kwargs:
        :return:
        """
        texts = []
        i = 0
        references = {}
        for obj, _, obj_meta in self.kb_adapter.search(
                query, relevance_factor=self.relevance_factor, **kwargs
        ):
            i += 1
            obj_text = yaml.dump({k: v for k, v in obj.items() if v}, sort_keys=False)
            references[str(i)] = obj_text
            texts.append(f"## Reference {i}\n{obj_text}")
        model = self.extractor.model
        prompt = f"Background facts:\n"
        prompt += "\n".join(texts)
        prompt += "\n\n"
        prompt += "I will ask a question and you will answer as best as possible, citing the references above.\n"
        prompt += "Write references in square brackets, e.g. [1].\n"
        prompt += "For additional facts you are sure of but a reference is not found, write [?].\n"
        prompt += f"---\nQuestion: {query}.\n"
        logger.info(f"Prompt: {prompt}")
        if conversation:
            conversation.model = model
            agent = conversation
            conversation_id = conversation.id
            logger.info(f"Conversation ID: {conversation_id}")
        else:
            agent = model
            conversation_id = None
        response = agent.prompt(prompt, system="You are a scientist assistant.")
        response_text = response.text()
        pattern = r'\[(\d+|\?)\]'
        used_references = re.findall(pattern, response_text)
        used_references_dict = {ref: references.get(ref, "NO REFERENCE") for ref in used_references}
        formatted_text = replace_references_with_links(response_text)
        return ChatResponse(response=response_text,
                            formatted_response=formatted_text,
                            prompt=prompt, references=used_references_dict, conversation_id=conversation_id)
