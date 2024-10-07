from curategpt.cli import main
from tests import INPUT_DIR

ONT_DB = str(INPUT_DIR / "go-nucleus.db")


def test_store_management(runner):
    result = runner.invoke(
        main, ["ontology", "index", ONT_DB, "-D", "chromadb", "-m", "all-MiniLM-L6-v2", "-c", "oai"]
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
    assert "default" in result.output
    assert "oai" in result.output
    assert "OntologyClass" in result.output
    result = runner.invoke(
        main, ["collections", "set", "-c", "default", "description: test description"]
    )
    assert result.exit_code == 0
