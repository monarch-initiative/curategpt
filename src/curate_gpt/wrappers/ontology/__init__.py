from pathlib import Path
from .ontology_wrapper import OntologyWrapper

THIS_DIR = Path(__file__).parent
ONTOLOGY_MODEL_PATH = THIS_DIR / "ontology.yaml"

__all__ = [
    "OntologyWrapper"
]
