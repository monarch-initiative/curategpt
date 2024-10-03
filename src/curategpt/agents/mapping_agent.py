"""Chat with a KB."""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from random import shuffle
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import inflection
import yaml
from curategpt.agents.base_agent import BaseAgent
from curategpt.formatters.format_utils import remove_formatting
from curategpt.store.db_adapter import SEARCH_RESULT
from curategpt.utils.tokens import estimate_num_tokens, max_tokens_by_model
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Which of the following entities best matches the concept "{query}"?
---
{body}
---
Give the result as a json list of references, for example
[2, 3]. If there are no matches, return an empty list.
Return only matches where you are confident the match is the same.


Concept to match: "{query}"
Answer:
"""


class MappingPredicate(str, Enum):
    SAME_AS = "SAME_AS"
    CLOSE_MATCH = "CLOSE_MATCH"
    BROAD_MATCH = "BROAD_MATCH"
    NARROW_MATCH = "NARROW_MATCH"
    RELATED_MATCH = "RELATED_MATCH"
    DIFFERENT_FROM = "DIFFERENT_FROM"
    UNKNOWN = "UNKNOWN"


class Mapping(BaseModel):
    """Response from chat engine."""

    model_config = ConfigDict(protected_namespaces=())
    subject_id: str
    object_id: str
    predicate_id: Optional[MappingPredicate] = None


class MappingSet(BaseModel):

    model_config = ConfigDict(protected_namespaces=())
    mappings: List[Mapping]
    prompt: str = None
    response_text: str = None


@dataclass
class MappingAgent(BaseAgent):
    """
    An agent to map/align entities.
    """

    relevance_factor: float = 1.0
    """Relevance factor for diversifying search results using MMR.
    high is recommended for this task"""

    def match(
        self,
        query: Union[str, Dict[str, Any]],
        limit: int = None,
        randomize_order: bool = False,
        include_predicates: bool = False,
        fields: List[str] = None,
        id_field: str = "id",
        **kwargs,
    ) -> MappingSet:
        """
        Match entities

        :param query:
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
            self.knowledge_source.search(
                query, relevance_factor=self.relevance_factor, limit=limit, **kwargs
            )
        )
        if randomize_order:
            # primarily for testing
            shuffle(kb_results)
        if include_predicates:
            mappings = list(self.categorize_mappings(query, kb_results, **kwargs))
            return MappingSet(mappings=mappings)
        model = self.extractor.model
        while True:
            i = 0
            references = {}
            objects = {}
            texts = []
            current_length = 0
            for obj, _, _obj_meta in kb_results:
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

        # Need to remove Markdown formatting here or it won't parse as JSON
        response_text = remove_formatting(text=response.text(), expect_format="json")

        mappings = []
        try:
            for m in json.loads(response_text):
                m = str(m)
                if m in objects:
                    object_id = objects.get(m)["id"]
                else:
                    object_id = m
                mappings.append(
                    Mapping(
                        subject_id=query,
                        object_id=object_id,
                        predicate=MappingPredicate.SAME_AS,
                        prompt=prompt,
                    )
                )
        except json.decoder.JSONDecodeError:
            # This will happen if the response is still not valid JSON
            # This returns an empty set of mappings, but the prompt and response text are retained
            return MappingSet(mappings=mappings, prompt=prompt, response_text=response_text)

        return MappingSet(mappings=mappings, prompt=prompt, response_text=response_text)

    def categorize_mappings(
        self, query: Union[str, Dict[str, Any]], kb_results: List[SEARCH_RESULT], **kwargs
    ) -> Iterator[Mapping]:
        """
        Categorize mappings predicate

        :param query:
        :param kb_results:
        :return:
        """
        for result in kb_results:
            prompt = (
                "I will give you two concepts, and your job is to tell me how they are related.\n\n"
            )
            prompt += f"Allowed answers are {', '.join(MappingPredicate.__members__)}:\n"
            prompt += "What is the relationship between these two concepts:\n"
            prompt += f"Concept A:\n{query}\n"
            prompt += f"Concept B:\n{yaml.dump(result[0], sort_keys=False)}"
            prompt += "\n\n"
            prompt += "Relationship:\n"
            model = self.extractor.model
            response = model.prompt(prompt)
            response_text = response.text()
            if response_text in MappingPredicate.__members__:
                pred = MappingPredicate[response_text]
            else:
                pred = MappingPredicate.UNKNOWN
            m = Mapping(subject_id=query, object_id=result[0]["id"], predicate_id=pred)
            yield m

    def find_links(self, other_collection: str) -> Iterator[Tuple[str, str, str]]:
        """
        Find links between elements in this collection and another collection

        :param other_collection:
        :return:
        """
        # TODO
        for _, _, info in self.knowledge_source.find(
            collection=other_collection, include=["embeddings", "documents", "metadatas"]
        ):
            embeddings = info["embeddings"]
            self.knowledge_source.find(embeddings, limit=10)
