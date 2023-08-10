"""Retrieval Augmented Generation (RAG)."""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

import openai
import yaml
from pydantic import BaseModel as BaseModel

from curate_gpt.rag.rag import OBJECT, RAG
from curate_gpt.store.db_adapter import DBAdapter

FUNC_NAME = "extract_data"

logger = logging.getLogger(__name__)

# TODO: refactor to reuse extractor


@dataclass
class OpenAIRAG(RAG):
    max_tokens: int = 3000
    model: str = "gpt-3.5-turbo-0613"
    conversation: List[Dict[str, Any]] = None
    conversation_mode: bool = False
    relevance_factor: float = 0.5

    def functions(self):
        return [
            {
                "name": FUNC_NAME,
                "description": "A n ontology term",
                "parameters": self.root_class.schema(),
            },
        ]

    def generate(self, text: str, num_examples=3, mode=1, **kwargs) -> OBJECT:
        """
        Generate a structured object from text.

        :param text:
        :param conversation_mode:
        :return:
        """
        conversation_mode = self.conversation_mode
        examples = list(
            self.db_adapter.search(text, relevance_factor=self.relevance_factor, **kwargs)
        )

        # TODO
        kind = "Ontology Term"
        text_field = "label"

        messages = [
            {"role": "system", "content": f"You are system that returns {kind} object in JSON."},
        ]
        for example, _, __ in examples[0:num_examples]:
            ex_text = example[text_field]
            print(f"EXAMPLE = {ex_text}")
            messages.append(
                {
                    "role": "user",
                    "content": f"make terms for {ex_text}",
                }
            )
            if mode == 1:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": FUNC_NAME,
                            "arguments": json.dumps(example),
                        },
                    },
                )
            else:
                messages.append(
                    {
                        "role": "function",
                        "name": FUNC_NAME,
                        "content": json.dumps(example),
                    },
                )
        if conversation_mode and self.conversation:
            messages.extend(self.conversation)
        # content = f"make terms for {text}"
        content = text
        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )
        print(yaml.dump(messages))
        response = openai.ChatCompletion.create(
            model=self.model,
            functions=self.functions(),
            messages=messages,
            max_tokens=self.max_tokens,
        )
        logger.debug(f"RESPONSE = {response}")
        print(response)
        choice = response.choices[0]
        message = choice["message"]
        r = message["function_call"]["arguments"]
        obj = json.loads(r)
        if conversation_mode:
            if self.conversation is None:
                self.conversation = []
            self.conversation.append(messages[-1])
            self.conversation.append({"role": "function", "name": FUNC_NAME, "content": r})
        return obj
