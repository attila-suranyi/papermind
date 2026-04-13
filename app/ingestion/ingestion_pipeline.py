import logging
from pathlib import Path
from typing import List, Optional

from app.embedder import Embedder
from app.ingestion.ingest_pdf import ingest_pdf, ingest_pdfs
from app.model import Chunk
from app.store.chroma_db import ChromaDB
from config import DOCS_DIR

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, db: ChromaDB, embedder: Embedder):
        self.embedder = embedder
        self.db = db

    def index_pdfs(self, docs_dir: Optional[Path] = DOCS_DIR):
        """Ingest PDFs from `docs_dir`, generate embeddings, and index them.

        This method reads PDFs (via `ingest_pdfs`), produces chunk-level
        embeddings, and writes them into the configured vector DB collection.
        """
        try:
            ingested_pdfs = ingest_pdfs(docs_dir)
            if not ingested_pdfs:
                logger.info("No chunks returned")
                return

            logger.info("Successfully ingested %d PDF(s)", len(ingested_pdfs))
        except Exception:
            logger.exception("Failed to ingest PDFs")
            return

        for pdf_name, chunks in ingested_pdfs.items():
            self._index_chunks(pdf_name, chunks)

    def index_pdf(self, pdf_path: Path):
        """Ingest a single PDF, generate embeddings, and index it."""
        try:
            chunks = ingest_pdf(pdf_path)
            if not chunks:
                logger.info("No chunks returned for %s", pdf_path.name)
                return

            logger.info("Successfully ingested PDF: %s", pdf_path.name)
        except Exception:
            logger.exception("Failed to ingest PDF: %s", pdf_path.name)
            return

        self._index_chunks(pdf_path.name, chunks)

    def _index_chunks(self, pdf_name: str, chunks: List[Chunk]):
        failed = 0
        successful = 0
        for i, chunk in enumerate(chunks):
            chunk_id = f"{pdf_name}_chunk_{i}"

            if not chunk.text:
                continue
            try:
                chunk.embedding = self.embedder.embed(chunk.text)
            except Exception:
                logger.exception("Failed to generate embedding for chunk %s", chunk_id)
                failed += 1
                continue

            try:
                self.db.add_chunk(chunk_id, chunk)
            except Exception as e:
                failed += 1
                logger.error("Failed to add chunk %s, error: %s", chunk_id, e)
                continue

            successful += 1

        logger.info(
            "Successfully ingested %d chunk(s), %d failed(s) for %s", successful, failed, pdf_name
        )
