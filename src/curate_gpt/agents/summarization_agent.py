import logging
from dataclasses import dataclass
from typing import List

from curate_gpt.agents.base_agent import BaseAgent
from curate_gpt.wrappers import BaseWrapper

logger = logging.getLogger(__name__)


@dataclass
class SummarizationAgent(BaseAgent):
    """
    An agent to summarize entities

    AKA SPINDOCTOR/TALISMAN
    """

    def summarize(
        self,
        object_ids: List[str],
        description_field: str,
        name_field: str,
        strict: bool = False,
        system_prompt: str = None,
    ):
        """
        Summarize a list of objects.

        Example:
        -------
        >>> extractor = BasicExtractor()
        >>> wrapper = get_wrapper("alliance_gene")
        >>> agent = SummarizationAgent(wrapper, extractor=extractor)
        >>> gene_ids = ["HGNC:9221", "HGNC:11195", "HGNC:6348", "HGNC:7553"]
        >>> response = agent.summarize(
        ...               gene_ids,
        ...               name_field="symbol",
        ...               description_field="automatedGeneSynopsis",
        ...               system_prompt="What function do these genes have in common?",
        ...           )
        >>> print(response)

        :param object_ids:
        :param description_field:
        :param name_field:
        :param strict:
        :param system_prompt:
        :return:
        """
        if not description_field:
            raise ValueError("Must provide a description field")
        if not name_field:
            raise ValueError("Must provide a name field")
        knowledge_source = self.knowledge_source
        if isinstance(knowledge_source, BaseWrapper):
            object_iter = knowledge_source.objects(self.knowledge_source_collection, object_ids)
        else:
            object_iter = knowledge_source.lookup_multiple(
                object_ids, collection=self.knowledge_source_collection
            )
        objects = list(object_iter)
        descriptions = [
            (obj.get(name_field, ""), obj.get(description_field, None)) for obj in objects
        ]
        if any(desc[0] is None for desc in descriptions):
            raise ValueError(f"Missing name for objects: {objects}")
        if strict:
            missing_descriptions = [name for name, desc in descriptions if desc is None]
            if missing_descriptions:
                raise ValueError(f"Missing descriptions for objects: {missing_descriptions}")
        if not descriptions:
            raise ValueError(f"No descriptions found for objects: {objects}")
        desc_lengths = [len(desc[1] or "") for desc in descriptions]
        max_desc_length = max(desc_lengths)
        if max_desc_length > 30:
            sep = "\n---\n"
        else:
            sep = "; "
        text = sep.join([f"{desc[0]}: {desc[1]}" for desc in descriptions])
        logger.info(f"Summarizing {len(descriptions)} objects with {len(text)} characters")
        model = self.extractor.model
        if not system_prompt:
            system_prompt = "Summarize these entities"
        response = model.prompt(text, system=system_prompt)
        return response.text()
