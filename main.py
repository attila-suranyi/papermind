import logging

from app.ingestion.ingestion_pipeline import IngestionPipeline


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    ingestion_pipeline = IngestionPipeline()
    ingestion_pipeline.run()


if __name__ == "__main__":
    main()
