from pathlib import Path
from typing import Dict, List
import json

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from app.model.chunk import Chunk


def flatten_metadata(metadata: dict) -> dict:
    """
    Convert nested metadata to Chroma-compatible flat format.
    Complex objects (lists, dicts) are serialized as JSON strings.
    """
    flattened = {}
    
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            flattened[key] = value
        else:
            # Complex types (list, dict, etc.) are serialized as JSON strings
            flattened[key] = json.dumps(value)
    
    return flattened


def ingest_pdfs(dir_path: Path) -> Dict[str, List[Chunk]]:
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


def get_chunks(pdf_path: Path) -> List[Chunk]:
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
        chunks.append(Chunk(text=enriched_text, metadata=flatten_metadata(chunk.meta.export_json_dict())))

    print(f"Successfully ingested {len(chunks)} chunks")

    return chunks
