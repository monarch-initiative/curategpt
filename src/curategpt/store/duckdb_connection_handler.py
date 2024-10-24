import logging
import duckdb
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DuckDBConnectionAndRecoveryHandler:
    def __init__(self, path: str):
        self.path = self._setup_path(path)
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    @staticmethod
    def _setup_path(path: str) -> str:
        """Handle path setup logic."""
        if not path:
            path = "./db/db_file.duckdb"
        if os.path.isdir(path):
            path = os.path.join("./db", path, "db_file.duckdb")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info(
                f"Path {path} is a directory. Using {path} as the database path\n\
                as duckdb needs a file path"
            )
        return path

    @staticmethod
    def _kill_process(pid: int) -> None:
        """Kill a process if it's holding the database lock."""
        try:
            import psutil
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"Successfully terminated process {pid}")
        except Exception as e:
            logger.warning(f"Failed to kill process {pid}: {e}")

    @staticmethod
    def _load_vss_extensions(conn: duckdb.DuckDBPyConnection) -> None:
        """Load VSS extensions for a connection."""
        conn.execute("INSTALL vss;")
        conn.execute("LOAD vss;")
        conn.execute("SET hnsw_enable_experimental_persistence=true;")

    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Establish database connection with error handling and recovery.

        Workflow as described in:
        https://duckdb.org/docs/extensions/vss.html#persistence

        In case of any WAL related issue:
        - Create a temporary workspace (in-memory database with VSS)
        - Temporarily bring in the broken database (ATTACH)
        - Fix it (WAL recovery happens)
        - Save changes (CHECKPOINT)
        - Put the fixed database back (DETACH)
        - Clean up our temporary workspace (close)
        - Now safely open the fixed database normally

        """
        wal_path = Path(self.path + '.wal')
        if wal_path.exists():
            logger.info("Found WAL file, attempting recovery...")
            try:
                temp_conn = duckdb.connect(':memory:')
                self._load_vss_extensions(temp_conn)
                temp_conn.execute(f"ATTACH '{self.path}' AS main_db")
                temp_conn.execute("CHECKPOINT;")
                temp_conn.execute("DETACH main_db")
                temp_conn.close()
            except Exception as e:
                logger.warning(f"WAL recovery attempt failed: {e}")

        try:
            self.conn = duckdb.connect(self.path, read_only=False)

        except duckdb.Error as e:
            match = re.search(r"PID (\d+)", str(e))
            if match:
                pid = int(match.group(1))
                logger.info(f"Got {e}. Attempting to kill process with PID: {pid}")
                self._kill_process(pid)
                self.conn = duckdb.connect(self.path, read_only=False)
            else:
                logger.error(f"Connection error without PID information: {e}")
                raise

        self._load_vss_extensions(self.conn)

        return self.conn

    def close(self) -> None:
        """Safely close the database connection."""
        if self.conn:
            try:
                self.conn.execute("CHECKPOINT;")
                self.conn.close()
                logger.info("Database connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")