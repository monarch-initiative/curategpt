import logging
from typing import Dict, List, Optional, Tuple

import yaml
from llm import Model

from curate_gpt.store.db_adapter import SEARCH_RESULT
from curate_gpt.utils.tokens import estimate_num_tokens, max_tokens_by_model

logger = logging.getLogger(__name__)


def select_from_options_prompt(
    kb_results: List[SEARCH_RESULT],
    model: Model,
    obj_type: str = "Reference",
    query: Optional[str] = None,
    prompt_template: Optional[str] = None,
    id_field: str = None,
) -> Tuple[str, Dict[str, str], Dict]:
    """
    Prompt user to select from a list of options.

    :param kb_results: order from most relevant
    :param model:
    :param obj_type:
    :param query:
    :param prompt_template:
    :return:
    """
    if prompt_template is None:
        if query is None:
            raise ValueError("Either query or prompt_template must be specified.")
        prompt_template = "I will first give background facts, then ask a question."
        prompt_template += "Use the background fact to answer\n"
        prompt_template += "---\nBackground facts:\n"
        prompt_template += "\n {join(texts)}"
        prompt_template += "\n\n"
        prompt_template += "I will ask a question and you will answer as best as possible,"
        prompt_template += "citing the references above.\n"
        prompt_template += "Write references in square brackets, e.g. [1].\n"
        prompt_template += (
            "For additional facts you are sure of but a reference is not found, write [?].\n"
        )
        prompt_template += "---\nHere is the Question: {query}.\n"
    while True:
        i = 0
        references = {}
        objects = {}
        texts = []
        current_length = 0
        for obj, _, _obj_meta in kb_results:
            i += 1
            obj_text = yaml.dump({k: v for k, v in obj.items() if v}, sort_keys=False)
            references[str(i)] = obj_text
            objects[str(i)] = obj
            if id_field and id_field in obj:
                ref = obj[id_field]
            else:
                ref = f"{obj_type} {i}"
            texts.append(f"## {ref}\n{obj_text}")
            current_length += len(obj_text)
        prompt = prompt_template.format(body="".join(texts), query=query)
        logger.info(f"Prompt: {prompt}")
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
    return prompt, references, objects
