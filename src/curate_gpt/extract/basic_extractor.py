"""Basic Extractor that is purely example driven."""
import json
import logging
from dataclasses import dataclass
from typing import List

from .extractor import AnnotatedObject, Extractor

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
        **kwargs,
    ) -> AnnotatedObject:
        logger.debug(f"Basic extractor: {text}, {len(examples)} examples")
        prompt = ""
        if background_text:
            prompt += f"{background_text}\n\n"
        prompt += (
            f"Extract a {target_class} object from text in {self.serialization_format} format.\n\n"
        )
        if rules:
            prompt += "Rules:\n\n"
            for rule in rules:
                prompt += f"- {rule}\n"
            prompt += "\n"
            prompt += "---\n"
        prompt += "Examples:\n\n"
        for example in examples:
            prompt += f"##\nText: {example.text}\n"
            prompt += f"Response: {self.serialize(example)}\n"
        prompt += f"\n##\nText: {text}\n\n"
        prompt += "Response: "
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
            return AnnotatedObject(object=json.loads(text))
        except Exception as e:
            if self.raise_error_if_unparsable:
                raise e
            else:
                return AnnotatedObject(object={})
