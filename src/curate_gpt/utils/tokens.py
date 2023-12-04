# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
import logging
from typing import List, Optional

import tiktoken

logger = logging.getLogger(__name__)


def max_tokens_by_model(model_id: Optional[str] = None):
    """
    Return the maximum number of tokens allowed by a model.

    TODO: return precise values, currently an estimate.
    """
    if model_id == "gpt-4":
        return 8192
    elif model_id == "gpt-3.5-turbo-16k":
        return 16384
    else:
        return 4097


def estimate_num_tokens(messages: List[str], model="gpt-4"):
    """
    Return the number of tokens used by a list of messages.

    Note: this is an estimate
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        # tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        # tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        logger.info(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return estimate_num_tokens(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        logger.info(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return estimate_num_tokens(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"num_tokens_from_messages() is not implemented for model {model}."
            "See https://github.com/openai/openai-python/blob/main/chatml.md"
            " for information on how messages are converted to tokens."
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        num_tokens += len(encoding.encode(message))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
