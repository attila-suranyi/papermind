from app.ingestion.ingestion_pipeline import IngestionPipeline


def main():
    #TODO use logger instead of print

    ingestion_pipeline = IngestionPipeline()
    ingestion_pipeline.run()

if __name__ == "__main__":
    main()
