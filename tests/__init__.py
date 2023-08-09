"""Tests for curate-gpt."""
from pathlib import Path

this_directory = Path(__file__).resolve().parent
INPUT_DIR = this_directory / "input"
INPUT_DBS = INPUT_DIR / "dbs"
OUTPUT_DIR = this_directory / "output"
OUTPUT_CHROMA_DB_PATH = OUTPUT_DIR / "db"
