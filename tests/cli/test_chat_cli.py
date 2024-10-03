from curategpt.cli import main

from tests import INPUT_DIR
from tests.store.conftest import requires_openai_api_key

ONT_DB = str(INPUT_DIR / "go-nucleus.db")


@requires_openai_api_key
def test_chat_cli(runner):
    result = runner.invoke(
        main, ["ontology", "index", ONT_DB, "-m", "openai:", "-c", "oai", "-D", "chromadb"]
    )
    assert result.exit_code == 0
    result = runner.invoke(
        main,
        [
            "ask",
            "-c",
            "oai",
            "What is the section between the lipid bilayers of the nuclear envelope called?",
            "-D",
            "chromadb",
        ],
    )
    assert result.exit_code == 0
    print(result.output)
