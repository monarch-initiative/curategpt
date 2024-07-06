"""Tests for curate-gpt."""
import os
from pathlib import Path

this_directory = Path(__file__).resolve().parent
INPUT_DIR = this_directory / "input"
INPUT_DBS = INPUT_DIR / "dbs"
OUTPUT_DIR = this_directory / "output"
OUTPUT_CHROMA_DB_PATH = OUTPUT_DIR / "db"
OUTPUT_DUCKDB_PATH = os.path.join(OUTPUT_DIR, "duckdbvss.db")

