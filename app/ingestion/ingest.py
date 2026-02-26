from pathlib import Path
from typing import Dict, List

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter


def ingest_pdf(pdf_path: str) -> List[Dict]:
    p = Path(pdf_path)

    if not p.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    source = p.resolve()
    converter = DocumentConverter()

    doc = converter.convert(source).document

    chunker = HybridChunker()
    chunk_iter = chunker.chunk(dl_doc=doc)

    chunks = []

    for i, chunk in enumerate(chunk_iter):
        print(f"=== {i} ===")

        enriched_text = chunker.contextualize(chunk=chunk)
        chunks.append({"enriched_text": enriched_text, "metadata": chunk.meta.export_json_dict()})

        print(
            f"Chunk text:\n{chunk.text}\n, \
              enriched text:\n{enriched_text}\n, \
              metadata:\n{chunk.meta.export_json_dict()}\n"
        )

    return chunks
