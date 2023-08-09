"""Views that bridge a store to an external data model."""
from pathlib import Path

VIEW_DIR = Path(__file__).parent
"""Path to the view directory"""

ONTOLOGY_MODEL_PATH = VIEW_DIR / "ontology.yaml"
