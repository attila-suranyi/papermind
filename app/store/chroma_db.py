from pathlib import Path
from typing import Optional

import chromadb

from config import VECTOR_DB_DIR


class ChromaDB:

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = VECTOR_DB_DIR

        if not db_path.exists():
            db_path.mkdir(parents=True, exist_ok=True)

        if not db_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {str(db_path)}")

        self.client = chromadb.PersistentClient(path=str(db_path))
