from pathlib import Path
from typing import Dict, List

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter


def ingest_pdfs(dir_path: Path) -> Dict[str, List[Dict]]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {str(dir_path)}")
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {str(dir_path)}")
    
    pdf_files = list(dir_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {str(dir_path)}")
        return {}
    
    results = {}
    for pdf_file in sorted(pdf_files):
        print(f"\nIngesting {pdf_file.name}...")
        try:
            chunks = get_chunks(pdf_file)
            results[pdf_file.name] = chunks
            print(f"Successfully ingested {pdf_file.name}")
        except Exception as e:
            print(f"Error ingesting {pdf_file.name}: {e}")
    
    return results


def get_chunks(pdf_path: Path) -> List[Dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {str(pdf_path)}")

    source = pdf_path.resolve()
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
            f"Enriched text:\n{enriched_text}\n, \
              metadata:\n{chunk.meta.export_json_dict()}\n"
        )

    return chunks
