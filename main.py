from pathlib import Path
import chromadb
from app.embedder.embedder import Embedder
from app.ingestion.ingest import ingest_pdfs
from sentence_transformers import SentenceTransformer


def main():
    docs_dir = Path(__file__).parent / "docs"

    try:
        ingested_pdfs = ingest_pdfs(docs_dir)
        if not ingested_pdfs:
            print("No chunks returned")
            return

        print(f"\nSuccessfully ingested {len(ingested_pdfs)} PDF(s)")
    except Exception as e:
        print(f"Failed to ingest PDFs: {e}")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedder = Embedder(model)

    db_path = Path(__file__).parent / "vector_db"
    db_client = chromadb.PersistentClient(path=db_path)
    pdf_collection_name = "pdf_collection"
    collection = db_client.get_or_create_collection(name=pdf_collection_name)

    for pdf_name, chunks in ingested_pdfs.items():
        for i, chunk in enumerate(chunks):
            if not chunk.text:
                continue
            embedding = embedder.embed(chunk.text)
            chunk.embedding = embedding

            try:
                collection.add(ids=(pdf_name + "_chunk_" + str(i)),
                           embeddings=chunk.embedding,
                           documents=chunk.text,
                           metadatas=chunk.metadata)
            except ValueError as e:
                print(e)


if __name__ == "__main__":
    main()
