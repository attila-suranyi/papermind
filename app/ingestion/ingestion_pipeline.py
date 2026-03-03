import logging
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from app.embedder import Embedder
from app.ingestion.ingest_pdf import ingest_pdfs
from app.store.chroma_db import ChromaDB
from config import DOCS_DIR, EMBEDDER_MODEL, VECTOR_DB_COLLECTION_NAME

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, db: Optional[ChromaDB] = None, model_name=EMBEDDER_MODEL):
        model = SentenceTransformer(model_name)
        self.embedder = Embedder(model)
        self.db = db or ChromaDB()

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

        collection = self.db.client.get_or_create_collection(name=VECTOR_DB_COLLECTION_NAME)

        failed = 0
        successful = 0
        for pdf_name, chunks in ingested_pdfs.items():
            for i, chunk in enumerate(chunks):
                chunk_id = pdf_name + "_chunk_" + str(i)

                if not chunk.text:
                    continue
                try:
                    embedding = self.embedder.embed(chunk.text)
                except ValueError:
                    logger.exception("Failed to generate embedding for chunk %s", chunk_id)
                    failed += 1
                    continue

                chunk.embedding = embedding

                try:
                    collection.add(
                        ids=[chunk_id],
                        embeddings=[chunk.embedding],
                        documents=[chunk.text],
                        metadatas=[chunk.metadata],
                    )
                except ValueError as e:
                    failed += 1
                    logger.error("Failed to add chunk #%d, error: %s", i, e)
                    continue

                successful += 1

        logger.info("Successfully ingested %d chunk(s), %d failed(s)", successful, failed)
