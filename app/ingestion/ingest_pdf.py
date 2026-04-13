import json
import logging
from pathlib import Path
from typing import Dict, List

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from app.model import Chunk

logger = logging.getLogger(__name__)


def ingest_pdfs(dir_path: Path) -> Dict[str, List[Chunk]]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {str(dir_path)}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {str(dir_path)}")

    pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        logger.info("No PDF files found in %s", str(dir_path))
        return {}

    results = {}
    for pdf_file in sorted(pdf_files):
        logger.info("Ingesting %s...", pdf_file.name)
        try:
            chunks = ingest_pdf(pdf_file)
            results[pdf_file.name] = chunks
            logger.info("Successfully ingested %s", pdf_file.name)
        except Exception:
            logger.exception("Error ingesting %s", pdf_file.name)

    return results


def ingest_pdf(pdf_path: Path) -> List[Chunk]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {str(pdf_path)}")

    source = pdf_path.resolve()
    converter = DocumentConverter()
    doc = converter.convert(source).document

    chunker = HybridChunker()
    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = []

    for i, chunk in enumerate(chunk_iter):
        enriched_text = chunker.contextualize(chunk=chunk)
        chunks.append(
            Chunk(
                text=enriched_text,
                metadata=flatten_metadata(chunk.meta.export_json_dict()),
            )
        )

    logger.info("Successfully ingested %d chunks", len(chunks))

    return chunks


def flatten_metadata(metadata: dict) -> dict:
    flattened = {}

    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            flattened[key] = value
        else:
            flattened[key] = json.dumps(value)

    return flattened
