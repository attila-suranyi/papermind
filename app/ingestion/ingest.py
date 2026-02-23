from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter


def ingest_pdf(pdf_path: str) -> Any:
    p = Path(pdf_path)

    if not p.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    source = p.resolve()
    converter = DocumentConverter()
    
    result = converter.convert(source)
    doc = getattr(result, "document", result)
    
    return doc
