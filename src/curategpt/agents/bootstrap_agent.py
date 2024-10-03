from dataclasses import dataclass
from typing import Dict

import yaml
from curategpt.agents.base_agent import BaseAgent
from curategpt.conf.prompts import PROMPTS_DIR
from curategpt.extract import AnnotatedObject
from jinja2 import Template
from pydantic import BaseModel, ConfigDict


class KnowledgeBaseSpecification(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    kb_name: str
    description: str
    attributes: str
    main_class: str


@dataclass
class BootstrapAgent(BaseAgent):

    def bootstrap_schema(self, specification: KnowledgeBaseSpecification) -> AnnotatedObject:
        """
        Bootstrap a schema for a knowledge base.

        :param specification: Specification for the knowledge base.
        :return:
        """
        # use a jinja template
        template = Template(open(PROMPTS_DIR / "bootstrap-schema.j2").read())
        prompt = template.render(**specification.model_dump())
        extractor = self.extractor
        model = extractor.model
        response = model.prompt(prompt)
        ao = extractor.deserialize(response.text(), format="yaml")
        return ao

    def bootstrap_data(
        self, specification: KnowledgeBaseSpecification = None, schema: Dict = None
    ) -> str:
        """
        Bootstrap data for a knowledge base.

        :param specification: Specification for the knowledge base.
        :param schema: Schema for the knowledge base.
        :return:
        """
        spec_str = None
        if specification is not None:
            preamble = "Use the following knowledge base schema specification"
            spec_str = yaml.dump(specification.model_dump(), sort_keys=False)
        if schema is not None:
            preamble = (
                "Make sure the object you create is consistent with the following linkml schema"
            )
            spec_str = yaml.dump(schema, sort_keys=False)
        if not spec_str:
            raise ValueError("No specification or schema provided")
        extractor = self.extractor
        model = extractor.model
        prompt = (
            "Generate an example instance data objects in YAML, inside a ```yaml...``` block\n"
            "Choose any objects you like, but make them representative of the domain, "
            "with rich comprehensive information\n"
            "Separate each element with '---' breaks, such that each entry is a separate object\n"
            "Do not include the root container class"
        )
        prompt += f"{preamble}:\n{spec_str}"
        response = model.prompt(prompt)
        txt = response.text()
        if "```" in txt:
            txt = txt.split("```")[1]
            txt = txt.strip()
            if txt.startswith("yaml"):
                txt = txt[4:]
                txt = txt.strip()
        return txt
