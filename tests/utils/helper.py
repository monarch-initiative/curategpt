from pathlib import Path

from src.curategpt.store import ChromaDBAdapter

DEBUG_MODE = False

def create_db_dir(tmp_path, out_dir) -> Path:
    """Creates a temporary directory or uses the provided debug directory."""
    if DEBUG_MODE:
        temp_dir = out_dir
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    else:
        return tmp_path


def setup_db(temp_dir: Path) -> ChromaDBAdapter:
    """Sets up the DBAdapter and optionally resets it."""
    # TODO: for now ChromaDB, later add DuckDB
    # db = get_store("chromadb", str(temp_dir))
    db = ChromaDBAdapter(str(temp_dir))
    # reset only when we use the db in try block, or in test
    return db

