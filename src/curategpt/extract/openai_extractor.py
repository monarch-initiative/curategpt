"""Extractor that uses OpenAI functions."""

import json
import logging
import os
from dataclasses import dataclass
from typing import List

from curate_gpt.extract.extractor import AnnotatedObject, Extractor
from openai import OpenAI, OpenAIError

FUNC_NAME = "extract_data"

logger = logging.getLogger(__name__)


@dataclass
class OpenAIExtractor(Extractor):
    """
    Extractor that uses OpenAI functions.
    """

    max_tokens: int = 3000
    model: str = "gpt-4"
    # conversation: List[Dict[str, Any]] = None
    # conversation_mode: bool = False

    @staticmethod
    def _get_openai_client():
        """
        Private method to get an instance of the OpenAI client.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise OpenAIError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )
        return OpenAI(api_key=api_key)

    def functions(self):
        return [
            {
                "name": FUNC_NAME,
                "description": "A n ontology term",
                "parameters": self.schema_proxy.json_schema(),
            },
        ]

    def extract(
        self,
        text: str,
        target_class: str,
        examples: List[AnnotatedObject] = None,
        examples_as_functions=False,
        conversation=None,
        **kwargs,
    ) -> AnnotatedObject:
        messages = [
            {
                "role": "system",
                "content": f"You are system that returns {target_class} object in JSON.",
            },
        ]
        for example in examples:
            ex_text = example.text
            ex_object = example.object
            # print(f"EXAMPLE = {ex_text}")
            messages.append(
                {
                    "role": "user",
                    "content": f"make terms for {ex_text}",
                }
            )
            if not examples_as_functions:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": FUNC_NAME,
                            "arguments": json.dumps(ex_object),
                        },
                    },
                )
            else:
                messages.append(
                    {
                        "role": "function",
                        "name": FUNC_NAME,
                        "content": json.dumps(ex_object),
                    },
                )
        if conversation:
            messages.extend(conversation)
        # content = f"make terms for {text}"
        content = text
        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )
        # print(yaml.dump(messages))
        client = self._get_openai_client()

        response = client.chat.completions.create(
            model=self.model,
            functions=self.functions(),
            messages=messages,
            max_tokens=self.max_tokens,
        )
        logger.debug(f"RESPONSE = {response}")
        # print(response)
        choice = response.choices[0]
        message = choice.message
        if "function_call" not in message:
            if self.raise_error_if_unparsable:
                raise ValueError("No function call in response")
            r = "{}"
        else:
            r = message["function_call"]["arguments"]
        try:
            obj = json.loads(r)
            if conversation:
                conversation.append(messages[-1])
                conversation.append({"role": "function", "name": FUNC_NAME, "content": r})
        except json.decoder.JSONDecodeError as e:
            if self.raise_error_if_unparsable:
                raise e
            obj = {}
        return AnnotatedObject(object=obj)
