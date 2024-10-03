import re

import pytest
from curategpt.agents.chat_agent import ChatAgent
from curategpt.extract import BasicExtractor

from tests.store.conftest import requires_openai_api_key


@requires_openai_api_key
@pytest.mark.parametrize("limit", [100, 10])
def test_chat(go_test_chroma_db, limit):
    """Tests asking questions over a knowledge base."""
    chat = ChatAgent(knowledge_source=go_test_chroma_db, extractor=BasicExtractor())
    response = chat.chat(
        "what is the definition of a nuclear membrane?", limit=limit, collection="test"
    )
    print(response.prompt)
    assert "lipid bilayer" in response.body.lower()
    assert "lipid bilayer" in response.formatted_body.lower()
    assert re.match(r".*\[\d*1\d*].*", response.body)
    assert "NuclearMembrane" in response.references["1"]
