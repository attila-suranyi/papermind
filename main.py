import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from app.ingestion.ingestion_pipeline import IngestionPipeline
from app.retrieval.retrieval_pipeline import RetrievalPipeline


def main(argv: Optional[list] = None):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Index pdfs or answer query")
    parser.add_argument(
        "-m",
        "--mode",
        choices=("index", "answer"),
        required=True,
        help="Index pdfs or answer a query",
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Query which should be answered",
    )
    parser.add_argument(
        "--docs-dir",
        help="Optional path to the docs directory to index (overrides config.DOCS_DIR)",
    )

    args = parser.parse_args(argv)

    if args.mode == "index":
        docs_dir = Path(args.docs_dir) if args.docs_dir else None
        ingestion_pipeline = IngestionPipeline()
        if docs_dir:
            ingestion_pipeline.index_pdfs(docs_dir=docs_dir)
        else:
            ingestion_pipeline.index_pdfs()
        return 0

    if args.mode == "answer":
        if not args.query:
            parser.error("--query is required when mode is 'answer'")
        retrieval_pipeline = RetrievalPipeline()
        result = retrieval_pipeline.get_answer(args.query)
        if result is not None:
            print(result)
        return 0

    return None


if __name__ == "__main__":
    sys.exit(main())
