from pathlib import Path

from app.embedder.embedder import Embedder
from app.ingestion.ingest import ingest_pdfs
from sentence_transformers import SentenceTransformer


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedder = Embedder(model)

    docs_dir = Path(__file__).parent / "docs"
    try:
        ingested_pdfs = ingest_pdfs(docs_dir)
        if not ingested_pdfs:
            print("No chunks returned")
            return

        print(f"\nSuccessfully ingested {len(ingested_pdfs)} PDF(s)")

        for pdf_name, chunks in ingested_pdfs.items():
            for chunk in chunks:
                if not chunk.text:
                    continue
                embedding = embedder.embed(chunk.text)
                chunk.embedding = embedding
    except Exception as e:
        print(f"Failed to ingest PDFs: {e}")
        return


if __name__ == "__main__":
    main()
