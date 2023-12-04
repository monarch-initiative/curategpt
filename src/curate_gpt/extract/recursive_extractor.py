"""Basic Extractor that is purely example driven."""
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

from curate_gpt.extract.extractor import AnnotatedObject, Extractor

logger = logging.getLogger(__name__)


@dataclass
class RecursiveExtractor(Extractor):

    """
    Extractor that recursively extracts objects from text.

    See SPIRES
    """

    serialization_format: str = "json"
    model_name: str = "gpt-3.5-turbo"

    def extract(
        self,
        text: str,
        target_class: str,
        examples: List[AnnotatedObject] = None,
        path=None,
        **kwargs,
    ) -> AnnotatedObject:
        if path is None:
            path = []
        if path:
            pathstr = f"Path: {'/'.join(path)}\n"
        else:
            pathstr = ""
        sv = self.schemaview
        prompt = (
            f"Extract a {target_class} object from text in {self.serialization_format} format,\n"
        )
        prompt += "Conforming to the following schema:\n"
        for slot in sv.class_induced_slots(target_class):
            desc = f"## {slot.description}" if slot.description else ""
            if slot.range:
                if slot.range == "string":
                    rng = '"<text>"'
                else:
                    rng = f"<{slot.range}>"
            else:
                rng = "<VALUE>"
            if slot.multivalued:
                rng = f"[{rng}, ...]"
            prompt += f"{slot.name}: {rng} {desc}\n"
        if examples:
            prompt += "Examples:\n\n"
            for example in examples:
                prompt += f"##\nText: {example.text}\n"
                prompt += pathstr
                prompt += f"Response: {self.partially_serialize(example.object, path)}\n"
        prompt += f"\n##\nText: {text}\n\n"
        prompt += pathstr
        prompt += "Response: "
        model = self.model
        print(f"Prompt: {prompt}")
        response = model.prompt(prompt)
        partial_object = self.deserialize(response.text())
        print(f"PO: {partial_object}")
        for slot in sv.class_induced_slots(target_class):
            if slot.range not in sv.all_classes():
                continue
            v = partial_object.object.get(slot.name, None)
            if v:
                if isinstance(v, list):
                    vs = v
                else:
                    vs = [v]
                sub_objects = [
                    self.extract(
                        x, target_class=slot.range, examples=examples, path=path + [slot.name]
                    )
                    for x in vs
                ]
                if isinstance(v, list):
                    partial_object.object[slot.name] = [
                        sub_object.object for sub_object in sub_objects
                    ]
                else:
                    partial_object.object[slot.name] = sub_objects[0].object
        return AnnotatedObject(object=partial_object.object, annotations={"text": text})

    def partially_serialize(self, object: Any, path: List[str]) -> str:
        if isinstance(object, list):
            return "[" + ", ".join([self.partially_serialize(o, path) for o in object]) + "]"
        if len(path) == 0:
            partial_object = deepcopy(object)
            for k, v in partial_object.items():
                if isinstance(v, list):
                    if len(v) > 0 and isinstance(v[0], dict):
                        partial_object[k] = [self._dict_as_str(o) for o in v]
                elif isinstance(v, dict):
                    partial_object[k] = self._dict_as_str(v)
            return json.dumps(partial_object)
        else:
            nxt = path[0]
            return self.partially_serialize(object[nxt], path[1:])

    def _dict_as_str(self, d: Dict[str, Any]) -> str:
        return ", ".join([f"{k} is {v}" for k, v in d.items()])

    def deserialize(self, text: str) -> AnnotatedObject:
        print(f"Parsing: {text}")
        obj = json.loads(text)
        print(f"Direct object: {obj}")
        ao = AnnotatedObject(object=obj)
        print(f"AO object: {ao.object}")
        return ao
