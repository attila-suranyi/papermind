from pathlib import Path

from app.ingestion.ingest import ingest_pdfs


def main():
    docs_dir = Path(__file__).parent / "docs"
    try:
        results = ingest_pdfs(docs_dir)
        if results:
            print(f"\nSuccessfully ingested {len(results)} PDF(s)")
            for pdf_name in results:
                print(f" - {pdf_name}: {len(results[pdf_name])} chunks")
        return
    except Exception as e:
        print(f"Failed to ingest PDFs: {e}")
        return


if __name__ == "__main__":
    main()
