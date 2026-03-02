from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from app.embedder import Embedder
from app.ingestion.ingest_pdf import ingest_pdfs
from app.store.chroma_db import ChromaDB
from config import DOCS_DIR, EMBEDDER_MODEL, VECTOR_DB_COLLECTION_NAME


class IngestionPipeline:

    def __init__(self, db: Optional[ChromaDB] = None, model_name = EMBEDDER_MODEL ):
        model = SentenceTransformer(model_name)
        self.embedder = Embedder(model)
        self.db = db or ChromaDB()

    def run(self, docs_dir: Optional[Path] = DOCS_DIR):
        try:
            ingested_pdfs = ingest_pdfs(docs_dir)
            if not ingested_pdfs:
                print("No chunks returned")
                return

            print(f"\nSuccessfully ingested {len(ingested_pdfs)} PDF(s)")
        except Exception as e:
            print(f"Failed to ingest PDFs: {e}")
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
                except Exception as e:
                    print(f"Failed to generate embedding for chunk {chunk_id}, error: {e}")
                    failed += 1
                    continue

                chunk.embedding = embedding

                try:
                    #TODO use batching: https://cookbook.chromadb.dev/strategies/batching/
                    collection.add(
                        ids=[chunk_id],
                        embeddings=[chunk.embedding],
                        documents=[chunk.text],
                        metadatas=[chunk.metadata],
                    )
                except ValueError as e:
                    failed += 1
                    print(f"Failed to add chunk #{i}, error {e}")
                    continue

                successful += 1

        print(f"\nSuccessfully ingested {successful} chunk(s), {failed} failed(s)")