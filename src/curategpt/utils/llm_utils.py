"""Utilities for interacting with LLM APIs."""

import logging

from llm import Model, Response
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)


def is_rate_limit_error(exception):
    # List of fully qualified names of RateLimitError exceptions from various libraries
    rate_limit_errors = [
        "openai.error.RateLimitError",
        "openai.RateLimitError",
        "anthropic.error.RateLimitError",
        "anthropic.RateLimitError",
        # Add more as needed
    ]
    exception_full_name = f"{exception.__class__.__module__}.{exception.__class__.__name__}"
    logger.warning(f"Exception_full_name: {exception_full_name}")
    logger.warning(f"Exception: {exception}")
    return exception_full_name in rate_limit_errors


@retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
)
def query_model(model: Model, *args, **kwargs) -> Response:
    logger.debug(f"Querying model {model.model_id}, args: {args}, kwargs: {kwargs}")
    response = model.prompt(*args, **kwargs)
    _text = response.text()
    return response
