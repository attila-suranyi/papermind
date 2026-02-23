import sys
from app.ingestion.ingest import ingest_pdf


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path-to-pdf>")
        return
    pdf_path = sys.argv[1]
    try:
        doc = ingest_pdf(pdf_path)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Failed to ingest PDF: {e}")
        return

    try:
        print(doc.export_to_markdown())
    except Exception:
        print("Document ingested but export to markdown failed.")


if __name__ == "__main__":
    main()