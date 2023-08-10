import pytest
from click.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    runner = CliRunner()
    return runner