from pathlib import Path


def _find_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Project root not found")

PROJECT_ROOT = _find_root()
DOCS_DIR = PROJECT_ROOT / "docs"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB_COLLECTION_NAME = "pdf_collection"