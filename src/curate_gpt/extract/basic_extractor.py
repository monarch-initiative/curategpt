"""Basic Extractor that is purely example driven."""
import json
import logging
import re
from copy import copy
from dataclasses import dataclass
from typing import List

from .extractor import AnnotatedObject, Extractor
from ..utils.tokens import estimate_num_tokens, max_tokens_by_model

logger = logging.getLogger(__name__)


@dataclass
class BasicExtractor(Extractor):
    """
    Extractor that is purely example driven.
    """

    serialization_format: str = "json"
    model_name: str = "gpt-3.5-turbo"

    def extract(
        self,
        text: str,
        target_class: str,
        examples: List[AnnotatedObject] = None,
        background_text: str = None,
        rules: List[str] = None,
        min_examples=1,
        **kwargs,
    ) -> AnnotatedObject:
        logger.debug(f"Basic extractor: {text}, {len(examples)} examples")
        examples = copy(examples)
        while True:
            prompt = ""
            if background_text:
                prompt += f"{background_text}\n\n"
            prompt += f"Extract a {target_class} object from text in {self.serialization_format} format.\n\n"
            if rules:
                prompt += "Rules:\n\n"
                for rule in rules:
                    prompt += f"- {rule}\n"
                prompt += "\n"
                prompt += "---\n"
            prompt += "Examples:\n\n"
            for example in examples:
                if example.text:
                    prompt += f"##\nText: {example.text}\n"
                prompt += f"Response: {self.serialize(example)}\n"
            prompt += f"\n##\nText: {text}\n\n"
            prompt += "Response: "
            estimated_length = estimate_num_tokens([prompt])
            if estimated_length + 300 < max_tokens_by_model(self.model.model_id):
                break
            else:
                # remove least relevant
                logger.debug(f"Removing least relevant of {len(examples)}: {examples[-1]}")
                examples.pop()
                if len(examples) < min_examples:
                    raise ValueError(
                        f"Prompt too long, need at least {min_examples} examples: {prompt}."
                    )
        model = self.model
        logger.info(f"Prompt: {prompt}")
        response = model.prompt(prompt)
        ao = self.deserialize(response.text())
        ao.annotations["prompt"] = prompt
        return ao

    def serialize(self, ao: AnnotatedObject) -> str:
        return json.dumps(ao.object)

    def deserialize(self, text: str) -> AnnotatedObject:
        logger.debug(f"Parsing {text}")
        try:
            obj = json.loads(text)
            if isinstance(obj, str):
                if self.raise_error_if_unparsable:
                    raise ValueError(f"Could not parse {text}")
                else:
                    obj = {}
            return AnnotatedObject(object=obj)
        except Exception as e:
            match = re.search(r"\{.*\}", text)
            if match:
                if match.group() != text:
                    return self.deserialize(match.group())
            if self.raise_error_if_unparsable:
                raise e
            else:
                logger.warning(f"Could not parse {text}")
                return AnnotatedObject(object={})
