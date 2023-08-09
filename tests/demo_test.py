"""Demo version test."""

import unittest
from dataclasses import dataclass, field
from typing import Dict, ClassVar

from curate_gpt import __version__

@dataclass
class Foo:
    foo_meta: ClassVar = {"a": "b"}

    d: Dict[str, str] = field(default_factory=dict)
    n: int = 1


class TestVersion(unittest.TestCase):
    """Test version."""

    def test_version_type(self):
        """Demo test."""
        self.assertIsInstance(__version__, str)

    def test_foo(self):
        """Demo test."""
        foo1 = Foo()
        foo2 = Foo()
        foo1.d["a"] = "99"
        assert foo1.d["a"] == "99"
        assert "a" not in foo2.d
        assert foo1.n == 1
        assert foo2.n == 1
        foo1.n = 2
        assert foo1.n == 2
        assert foo2.n == 1
