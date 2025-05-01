"""Chat with a KB."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml
from llm import Conversation
from pydantic import BaseModel, ConfigDict

from curategpt.agents.base_agent import BaseAgent
from curategpt.utils.tokens import estimate_num_tokens, max_tokens_by_model
from curategpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    """
    Response from chat engine.

    TODO: Rename class to indicate that it is provenance-enabled chat
    """

    model_config = ConfigDict(protected_namespaces=())

    body: str
    """Text of response."""

    prompt: str
    """Prompt used to generate response."""

    formatted_body: str = None
    """Body formatted with markdown links to references."""

    references: Optional[Dict[str, Any]] = None
    """References for citations detected in response."""

    uncited_references: Optional[Dict[str, Any]] = None
    """Potential references for which there was no detected citation."""


def replace_references_with_links(text):
    """Replace references with links."""
    pattern = r"\[(\d+)\]"
    replacement = lambda m: f"[{m.group(1)}](#ref-{m.group(1)})"
    return re.sub(pattern, replacement, text)


@dataclass
class ChatAgent(BaseAgent):
    """
    An agent that allows chat to a knowledge source.

    This implements a standard knowledgebase retrieval augmented generation pattern.
    The knowledge_source is queried for relevant objects (the source can be a local
    database or a remote source such as pubmed).
    The objects are provided as context to a LLM query
    """

    relevance_factor: float = 0.5
    """Relevance factor for diversifying search results using MMR."""

    conversation_id: Optional[str] = None

    def chat(
        self,
        query: str,
        conversation: Optional[Conversation] = None,
        limit: int = 10,
        collection: str = None,
        expand=True,
        **kwargs,
    ) -> ChatResponse:
        """
        Extract structured object using text seed and background knowledge.

        :param text:
        :param kwargs:
        :return:
        """
        if self.extractor is None:
            if isinstance(self.knowledge_source, BaseWrapper):
                self.extractor = self.knowledge_source.extractor
            else:
                raise ValueError("Extractor must be set.")
        logger.info(f"Chat: {query} on {self.knowledge_source} kwargs: {kwargs}, limit: {limit}")
        if collection is None:
            collection = self.knowledge_source_collection
        kwargs["collection"] = collection
        kb_results = list(
            self.knowledge_source.search(
                query, relevance_factor=self.relevance_factor, limit=limit, expand=expand, **kwargs
            )
        )
        while True:
            i = 0
            references = {}
            texts = []
            current_length = 0
            for obj, _, _obj_meta in kb_results:
                i += 1
                obj_text = yaml.dump({k: v for k, v in obj.items() if v}, sort_keys=False)
                references[str(i)] = obj_text
                texts.append(f"## Reference {i}\n{obj_text}")
                current_length += len(obj_text)
            model = self.extractor.model
            prompt = "I will first give background facts, then ask a question. Use the background fact to answer\n"
            prompt += "---\nBackground facts:\n"
            prompt += "\n".join(texts)
            prompt += "\n\n"
            prompt += "I will ask a question and you will answer as best as possible, citing the references above.\n"
            prompt += "Write references in square brackets, e.g. [1].\n"
            prompt += (
                "For additional facts you are sure of but a reference is not found, write [?].\n"
            )
            prompt += f"---\nHere is the Question: {query}.\n"
            logger.debug(f"Candidate Prompt: {prompt}")
            estimated_length = estimate_num_tokens([prompt])
            logger.debug(
                f"Max tokens {self.extractor.model.model_id}: {max_tokens_by_model(self.extractor.model.model_id)}"
            )
            # TODO: use a more precise estimate of the length
            if estimated_length + 300 < max_tokens_by_model(self.extractor.model.model_id):
                break
            else:
                # remove least relevant
                logger.debug(f"Removing least relevant of {len(kb_results)}: {kb_results[-1]}")
                if not kb_results:
                    raise ValueError(f"Prompt too long: {prompt}.")
                kb_results.pop()

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
        pattern = r"\[(\d+|\?)\]"
        used_references = re.findall(pattern, response_text)
        used_references_dict = {ref: references.get(ref, "NO REFERENCE") for ref in used_references}
        uncited_references_dict = {
            ref: ref_obj for ref, ref_obj in references.items() if ref not in used_references
        }
        formatted_text = replace_references_with_links(response_text)
        return ChatResponse(
            body=response_text,
            formatted_body=formatted_text,
            prompt=prompt,
            references=used_references_dict,
            uncited_references=uncited_references_dict,
            conversation_id=conversation_id,
        )


@dataclass
class ChatAgentAlz(BaseAgent):
    """
    An agent that allows chat to a knowledge source.

    This implements a standard knowledgebase retrieval augmented generation pattern.
    The knowledge_source is queried for relevant objects (the source can be a local
    database or a remote source such as pubmed).
    The objects are provided as context to a LLM query
    """

    relevance_factor: float = 0.5
    """Relevance factor for diversifying search results using MMR."""

    conversation_id: Optional[str] = None

    def chat(
            self,
            query: str,
            conversation: Optional[Any] = None,
            limit: int = 10,
            collection: str = None,
            expand: bool = True,
            **kwargs,
    ) -> ChatResponse:
        if self.extractor is None:
            if isinstance(self.knowledge_source, BaseWrapper):
                self.extractor = self.knowledge_source.extractor
            else:
                raise ValueError("Extractor must be set.")

        logger.info(f"Chat: {query} on {self.knowledge_source} with limit: {limit}")
        if collection is None:
            collection = self.knowledge_source_collection
        kwargs["collection"] = collection

        # Set Alzheimer's system prompt if we are using paperqa
        if hasattr(self.knowledge_source, 'name') and self.knowledge_source.name == 'paperqa':
            self.knowledge_source.settings.agent.agent_system_prompt = (
                """You are a specialized AI assistant for biomedical researchers and clinicians focused on
                Alzheimer's disease and related topics. I will ask a question and you will answer
                as best as possible, citing references. For any additional facts that you are
                sure of, but without a citation, write [?].
                """)

        # The search now returns dictionary results directly.
        kb_results = self.knowledge_source.search(
            query,
            relevance_factor=self.relevance_factor,
            limit=limit,
            expand=expand,
            **kwargs
        )

        # Check if we're using PaperQA
        is_paperqa = hasattr(self.knowledge_source, 'name') and self.knowledge_source.name == 'paperqa'

        model = self.extractor.model
        if conversation:
            conversation.model = model
            agent = conversation
            conversation_id = conversation.id
            logger.info(f"Conversation ID: {conversation_id}")
        else:
            agent = model
            conversation_id = None

        if is_paperqa:
            session = kb_results.session
            response_text = session.answer.strip()
            prompt = f"[PaperQA] Question: {session.question}"
            formatted_body, references = (
                self._format_paperqa_references(response_text,
                                                session.contexts))

            # formatted_refs = {k: yaml.dump(v, sort_keys=False) for k, v in references.items() if v}

            def drop_empty_fields(d: dict) -> dict:
                return {k: v for k, v in d.items() if isinstance(v, str) and v.strip()}

            formatted_refs = {
                k: yaml.safe_dump(
                    drop_empty_fields(v),
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                    width=80  # prevents line wrapping madness
                )
                for k, v in references.items()
            }

            return ChatResponse(
                body=response_text,
                formatted_body=formatted_body,
                prompt=prompt,
                references=formatted_refs,
                uncited_references={},
                conversation_id=conversation_id,
            )

        else:
            kb_results = list(kb_results)
            # Regular processing for non-PaperQA sources

            # For other sources, we need to format the results and create a prompt
            while True:
                i = 0
                references = {}
                texts = []
                current_length = 0
                for obj, _, _obj_meta in kb_results:
                    i += 1
                    obj_text = yaml.dump({k: v for k, v in obj.items() if v}, sort_keys=False)
                    references[str(i)] = obj_text
                    texts.append(f"## Reference {i}\n{obj_text}")
                    current_length += len(obj_text)

                prompt = "I will first give background facts, then ask a question. Use the background fact to answer\n"
                prompt += "---\nBackground facts:\n"
                prompt += "\n".join(texts)
                prompt += "\n\n"
                prompt += "I will ask a question and you will answer as best as possible, citing the references above.\n"
                prompt += "Write references in square brackets, e.g. [1].\n"
                prompt += (
                    "For additional facts you are sure of but a reference is not found, write [?].\n"
                )
                prompt += f"---\nHere is the Question: {query}.\n"
                logger.debug(f"Candidate Prompt: {prompt}")
                estimated_length = estimate_num_tokens([prompt])
                logger.debug(
                    f"Max tokens {self.extractor.model.model_id}: {max_tokens_by_model(self.extractor.model.model_id)}"
                )
                # TODO: use a more precise estimate of the length
                if estimated_length + 300 < max_tokens_by_model(self.extractor.model.model_id):
                    break
                else:
                    # remove least relevant
                    logger.debug(f"Removing least relevant of {len(kb_results)}: {kb_results[-1]}")
                    if not kb_results:
                        raise ValueError(f"Prompt too long: {prompt}.")
                    kb_results.pop()

            logger.info(f"Prompt: {prompt}")

            response = agent.prompt(prompt, system="You are a scientist assistant.")
            response_text = response.text()
            pattern = r"\[(\d+)\]"
            used_references = re.findall(pattern, response_text)
            used_references_dict = {ref: references.get(ref, "NO REFERENCE") for ref in used_references}
            uncited_references_dict = {
                ref: ref_obj for ref, ref_obj in references.items() if ref not in used_references
            }
            formatted_text = replace_references_with_links(response_text)
            return ChatResponse(
                body=response_text,
                formatted_body=formatted_text,
                prompt=prompt,
                references=used_references_dict,
                uncited_references=uncited_references_dict,
                conversation_id=conversation_id,
            )

    @staticmethod
    def _format_paperqa_references(answer: str, contexts: list) -> tuple[str, dict]:
        import re
        from collections import OrderedDict

        formatted_body = answer
        doc_key_to_num = OrderedDict()
        references = {}

        # Assign numbers to unique doc.key
        for ctx in contexts:
            doc = ctx.text.doc
            if doc.key not in doc_key_to_num:
                doc_key_to_num[doc.key] = len(doc_key_to_num) + 1
                references[str(doc_key_to_num[doc.key])] = {
                    "id": doc.key if hasattr(doc, 'key') else "",
                    "title": doc.title if hasattr(doc, 'title') else "",
                    "abstract": doc.text if hasattr(doc, 'text') else "",
                    "citation": doc.citation if hasattr(doc, 'citation') else "",
                    "url": doc.doi_url if hasattr(doc, 'doi_url') else "",
                    "doi": doc.doi if hasattr(doc, 'doi') else ""
                }

        used_pairs = set()
        for ctx in contexts:
            text_name = ctx.text.name.strip()  # e.g. melendez2024 pages 6–7
            doc_key = ctx.text.doc.key
            ref_num = doc_key_to_num[doc_key]
            pages = text_name.split("pages")[
                -1].strip() if "pages" in text_name else None

            if (text_name, ref_num, pages) in used_pairs:
                continue
            used_pairs.add((text_name, ref_num, pages))

            markdown_link = f"[{ref_num}](#ref-{ref_num})"
            replacement = f"{markdown_link} (pages {pages})" if pages else markdown_link
            formatted_body = re.sub(re.escape(text_name), replacement,
                                    formatted_body)

        return formatted_body, references
