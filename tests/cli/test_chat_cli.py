from curate_gpt.cli import main

from tests import INPUT_DIR

ONT_DB = str(INPUT_DIR / "go-nucleus.db")


def test_chat_cli(runner):
    result = runner.invoke(main, ["ontology", "index", ONT_DB, "-m", "openai:", "-c", "oai"])
    assert result.exit_code == 0
    result = runner.invoke(
        main,
        [
            "ask",
            "-c",
            "oai",
            "What is the section between the lipid bilayers of the nuclear envelope called?",
        ],
    )
    assert result.exit_code == 0
    print(result.output)
