import logging
from typing import Any, Optional

from chromadb import QueryResult
from sentence_transformers import SentenceTransformer

from app.embedder import Embedder
from app.retrieval.prompt import get_prompt
from app.retrieval.retrieved_chunk import RetrievedChunk
from app.store.chroma_db import ChromaDB
from config import EMBEDDER_MODEL, VECTOR_DB_COLLECTION_NAME

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self, db: Optional[ChromaDB] = None, model_name=EMBEDDER_MODEL):
        model = SentenceTransformer(model_name)
        self.embedder = Embedder(model)
        self.db = db or ChromaDB()

    def get_answer(self, query: str):
        query_embedding = self.embedder.embed(query)
        collection = self.db.client.get_or_create_collection(VECTOR_DB_COLLECTION_NAME)

        try:
            # TODO omit results with low relevance / high distance
            query_result = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
            )
        except Exception:
            logger.exception("Failed to query collection")
            return None

        results = get_single_query_results(query_result)

        logger.info("Retrieved %d results for query", len(results))

        retrieved_chunks = []
        for result in results:
            retrieved_chunks.append(RetrievedChunk.from_query_result(result))

        prompt = get_prompt(query, retrieved_chunks)

        return prompt


def get_single_query_results(query_result: QueryResult) -> list[Any]:
    # The chroma client returns results in lists per query. We're only
    # sending a single query, so take the first element from each field.
    ids = query_result.get("ids", [[]])[0]
    distances = query_result.get("distances", [[]])[0]
    documents = query_result.get("documents", [[]])[0]
    metadatas = query_result.get("metadatas", [[]])[0]

    results = []
    for idx in range(len(ids)):
        results.append(
            {
                "id": ids[idx],
                "document": documents[idx] if idx < len(documents) else None,
                "metadata": metadatas[idx] if idx < len(metadatas) else None,
                "distance": distances[idx] if idx < len(distances) else None,
            }
        )
    return results
