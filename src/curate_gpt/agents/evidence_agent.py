from dataclasses import dataclass
from typing import Dict, Union

from curate_gpt.agents.base_agent import BaseAgent
from curate_gpt.agents.chat_agent import ChatAgent
from curate_gpt.formatters.format_utils import object_as_yaml
from curate_gpt.wrappers import BaseWrapper


@dataclass
class EvidenceAgent(BaseAgent):

    """
    An agent to find evidence for an object by querying a reference source.

    TODO: also extend to allow categorization of evidence type.
    """

    chat_agent: Union[ChatAgent, BaseWrapper] = None

    def find_evidence(self, obj: Union[str, Dict]):
        obj_as_str = obj if isinstance(obj, str) else object_as_yaml(obj)
        text = f"Find evidence in the given references to support an object: {obj_as_str}"
        return self.chat_agent.chat(text)
