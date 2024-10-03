import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import yaml
from curate_gpt import BasicExtractor
from curate_gpt.agents.base_agent import BaseAgent
from curate_gpt.agents.chat_agent import ChatAgent, ChatResponse
from curate_gpt.formatters.format_utils import object_as_yaml
from curate_gpt.utils.tokens import estimate_num_tokens, max_tokens_by_model
from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)


class EvidenceUpdatePolicyEnum(str, Enum):
    skip = "skip"
    append = "append"
    replace = "replace"


@dataclass
class EvidenceAgent(BaseAgent):
    """
    An agent to find evidence for an object by querying a reference source.

    The evidence agent is able to find (supporting and refuting) evidence for any of
    the following:

    - A simple statement in natural language
    - A simple structured dictionary object of key-value pairs
    - A complex structured dictionary object with nested key-value pairs

    The default source used is Pubmed, using :class:`PubmedWrapper` via :class:`ChatAgent`
    """

    chat_agent: Union[ChatAgent, BaseWrapper] = None

    evidence_update_policy: EvidenceUpdatePolicyEnum = field(default=EvidenceUpdatePolicyEnum.skip)

    def find_evidence(self, obj: Union[str, Dict]) -> ChatResponse:
        obj_as_str = obj if isinstance(obj, str) else object_as_yaml(obj)
        text = f"Find evidence in the given references to support an object: {obj_as_str}"
        return self.chat_agent.chat(text)

    def find_evidence_simple(self, query: str, limit: int = 10, **kwargs) -> Optional[List[Dict]]:
        logger.info(f"Finding evidence for: {query}")
        extractor = self.chat_agent.extractor
        if extractor is None:
            if isinstance(self.knowledge_source, BaseWrapper):
                extractor = self.knowledge_source.extractor
            if extractor is None:
                logger.warning("No extractor found. Using default.")
                extractor = BasicExtractor()
        chat_agent = self.chat_agent
        if isinstance(self.chat_agent, BaseWrapper):
            chat_agent = ChatAgent(
                knowledge_source=chat_agent,
                extractor=extractor,
            )
        collection = None
        # collection = self.chat_agent.knowledge_source_collection
        if collection is None:
            collection = self.knowledge_source_collection
        kwargs["collection"] = collection
        kb_results = list(
            chat_agent.knowledge_source.search(
                query, relevance_factor=chat_agent.relevance_factor, limit=limit, **kwargs
            )
        )
        # TODO: DRY
        while True:
            i = 0
            references = {}
            texts = []
            current_length = 0
            for obj, _, _obj_meta in kb_results:
                i += 1
                obj_text = yaml.dump({k: v for k, v in obj.items() if v}, sort_keys=False)
                references[str(i)] = obj_text
                texts.append(f"## Reference\n{obj_text}")
                current_length += len(obj_text)
            model = extractor.model
            logger.info(f"Using model: {model.model_id}")
            prompt = "---\nBackground literature:\n"
            prompt += "\n".join(texts)
            prompt += f"---\nHere is the Statement: {query}.\n"
            logger.debug(f"Candidate Prompt: {prompt}")
            estimated_length = estimate_num_tokens([prompt])
            logger.debug(
                f"Max tokens {extractor.model.model_id}: {max_tokens_by_model(extractor.model.model_id)}"
            )
            # TODO: use a more precise estimate of the length
            if estimated_length + 300 < max_tokens_by_model(extractor.model.model_id):
                break
            else:
                # remove least relevant
                logger.debug(f"Removing least relevant of {len(kb_results)}: {kb_results[-1]}")
                if not kb_results:
                    raise ValueError(f"Prompt too long: {prompt}.")
                kb_results.pop()

        logger.debug(f"Prompt: {prompt}")

        response = model.prompt(
            prompt,
            system="""
        You are a scientist assistant. Given a statement in subject-predicate-object form, your job is
        to look at the literature I provide you, and return an object that states whether the literature
        supports or refutes the statement.

        Return in YAML:

        ```
        - reference: IDENTIFIER  ## e.g PMID:123456
          supports: "SUPPORT" | "REFUTE" | "NO_EVIDENCE" | "PARTIAL" | "WRONG_STATEMENT"
          snippet: "EXCERPT"
          explanation: "<explanatory text, if necessary>"
        - ...
        ```

        Never put a space between the pubmed prefix and local ID. The correct way to write is PMID:123456.

        Do not include an entry if you cannot find it in the literature provided.

        Pay close attention to the predicate. For example, the following statement:

            Subject: The Sun Predicate: orbits Value: name: Mercury

        is false, because the subject and value are inverted. Use WRONG_STATEMENT in this case.

        Similarly, the following statement:

            Subject: Mercury Predicate: friend-of Value: name: The Sun

        Is false because the predicate is not correct for these two entities.
        Use WRONG_STATEMENT in this case.

        Additionally, if the statement is:

            Subject: Ebola Predicate: causes Value: {name: Bleeding, severity: Mild }

        And the literature says that Ebola causes SEVERE bleeding.
        Use WRONG_STATEMENT in this case.
        """,
        )
        response_text = response.text()
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            response_text = response_text.strip()
            if response_text.startswith("yaml"):
                response_text = response_text[4:].strip()
        try:
            return yaml.safe_load(response_text)
        except Exception as e:
            logger.error(f"Error parsing response: {response_text}: {e}")
            return None

    def find_evidence_complex(
        self, obj: Union[str, Dict], label_field=None, statement_fields: Optional[List] = None
    ) -> Dict:
        if not label_field:
            for candidates in ["name", "label", "code"]:
                if candidates in obj:
                    label_field = candidates
                    break
        if not label_field:
            raise ValueError(f"No label field found in object {obj}")
        label = obj.get(label_field, None)
        if not label:
            raise ValueError(f"No label found in object {obj}")
        new_evidences = []
        for k, v in obj.items():
            if k == label_field:
                continue
            if statement_fields and k not in statement_fields:
                continue

            def _add_evidence(input_obj: dict) -> dict:
                if "evidence" in input_obj:
                    if self.evidence_update_policy == EvidenceUpdatePolicyEnum.skip:
                        return input_obj
                q = f"Subject: {label} Value: {yaml.dump(input_obj)}"
                evidence = self.find_evidence_simple(q)
                if evidence:
                    if (
                        "evidence" in input_obj
                        and self.evidence_update_policy == EvidenceUpdatePolicyEnum.append
                    ):
                        input_obj["evidence"].append(evidence)
                    else:
                        input_obj["evidence"] = evidence
                    new_evidences.append(evidence)

            if isinstance(v, list):
                for sub_obj in v:
                    if isinstance(sub_obj, dict):
                        _add_evidence(sub_obj)
            elif isinstance(v, dict):
                _add_evidence(v)
        logger.info(f"Found {len(new_evidences)} evidence objects.")
        return obj
