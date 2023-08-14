"""Chat with a KB."""
import json
import logging
import re
from dataclasses import dataclass
from random import shuffle
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Union

import inflection
import yaml
from pydantic import BaseModel

from curate_gpt.agents.agent_utils import select_from_options_prompt
from curate_gpt.extract import AnnotatedObject, Extractor
from curate_gpt.store import DBAdapter
from curate_gpt.utils.tokens import estimate_num_tokens, max_tokens_by_model
from llm import Conversation

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Which of the following entities best matches the concept "{query}"?
---
{body}
---
Give the result as a json list of references, for example
[2, 3]. If there are no matches, return an empty list.


Concept to match: "{query}"
Answer:
"""


class Mapping(BaseModel):
    """Response from chat engine."""

    subject_id: str
    object_id: str
    predicate_id: Optional[str] = None


class MappingSet(BaseModel):
    mappings: List[Mapping]
    prompt: str
    response_text: str


@dataclass
class Mapper:
    """
    An agent to map/align entities
    """

    kb_adapter: DBAdapter = None
    """Adapter to structured knowledge base"""

    extractor: Extractor = None

    relevance_factor: float = 1.0
    """Relevance factor for diversifying search results using MMR.
    high is recommended for this task"""

    def match(
        self,
        query: Union[str, Dict[str, Any]],
        limit: int = None,
        randomize_order: bool = False,
        fields: List[str] = None,
        id_field: str = "id",
        **kwargs,
    ) -> MappingSet:
        """
        Match entities

        :param text:
        :param limit:
        :param randomize_order: randomize the order in which candidates are presented (mostly for testing purposes)
        :param kwargs:
        :return:
        """
        # TODO: this approach doesn't work well - better do do the MapperGPT approach
        # and evaluate each mapping individually
        if self.extractor is None:
            raise ValueError("Extractor must be set.")
        # de-camelcase-ify if the query is camelCase
        if re.match(r".*[a-z][A-Z]", query):
            query = inflection.titleize(query).lower()
        if limit is None:
            limit = 10
        kb_results = list(
            self.kb_adapter.search(
                query, relevance_factor=self.relevance_factor, limit=limit, **kwargs
            )
        )
        if randomize_order:
            shuffle(kb_results)
        model = self.extractor.model
        # prompt, _references, objects = select_from_options_prompt(kb_results, prompt_template=PROMPT_TEMPLATE, query=query, model=model)
        while True:
            i = 0
            references = {}
            objects = {}
            texts = []
            current_length = 0
            for obj, _, obj_meta in kb_results:
                i += 1
                obj_text = yaml.dump(
                    {k: v for k, v in obj.items() if v and (fields is None or k in fields)},
                    sort_keys=False,
                )
                references[str(i)] = obj_text
                objects[str(i)] = obj
                texts.append(f"## REF {i}\n{obj_text}")
                current_length += len(obj_text)
            prompt = PROMPT_TEMPLATE.format(body="".join(texts), query=query)
            logger.debug(f"Prompt: {prompt}")
            estimated_length = estimate_num_tokens([prompt])
            logger.debug(f"Max tokens {model.model_id}: {max_tokens_by_model(model.model_id)}")
            # TODO: use a more precise estimate of the length
            if estimated_length + 300 < max_tokens_by_model(model.model_id):
                break
            else:
                # remove least relevant
                if not kb_results:
                    raise ValueError(f"Prompt too long: {prompt}.")
                kb_results.pop()
        response = model.prompt(prompt)
        response_text = response.text()
        mappings = []
        for m in json.loads(response_text):
            m = str(m)
            if m in objects:
                object_id = objects.get(m)["id"]
            else:
                object_id = m
            mappings.append(Mapping(subject_id=query, object_id=object_id, prompt=prompt))
            # mappings.append(Mapping(subject_id=query, object_id=m["match"], predicate_id=m["type"], prompt=prompt))
        return MappingSet(mappings=mappings, prompt=prompt, response_text=response_text)
