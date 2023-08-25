from curate_gpt.cli import main
from tests import INPUT_DIR

ONT_DB = str(INPUT_DIR / "go-nucleus.db")


def test_store_management(runner):
    result = runner.invoke(
        main, ["ontology", "index", ONT_DB, "-m", "openai:", "-c", "oai"]
    )
    assert result.exit_code == 0
    result = runner.invoke(main, ["ontology", "index", ONT_DB, "-c", "default"])
    assert result.exit_code == 0
    result = runner.invoke(main, ["search", "-c", "default", "nuclear membrane"])
    assert result.exit_code == 0
    assert "nuclear membrane" in result.output
    result = runner.invoke(main, ["search", "-c", "oai", "nuclear membrane"])
    assert result.exit_code == 0
    assert "nuclear membrane" in result.output
    result = runner.invoke(main, ["collections", "list"])
    assert result.exit_code == 0
    print(result.output)
    assert "default" in result.output
    assert "oai" in result.output
    assert "OntologyClass" in result.output
    result = runner.invoke(
        main, ["collections", "set", "-c", "default", "description: test description"]
    )
    assert result.exit_code == 0
