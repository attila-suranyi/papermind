import logging

from app.embedder import Embedder
from app.llm import LLMClient
from app.retrieval.prompt import get_prompt
from app.store.chroma_db import ChromaDB
from config import LLM_MODEL

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self, db: ChromaDB, embedder: Embedder, llm: LLMClient):
        self.embedder = embedder
        self.db = db
        self.llm = llm

    def get_answer(self, query: str):
        query_embedding = self.embedder.embed(query)

        try:
            # TODO omit results with low relevance / high distance
            retrieved_chunks = self.db.query_chunks(query_embedding, n_results=5)
        except Exception:
            logger.exception("Failed to query collection")
            return None

        logger.info("Retrieved %d results for query", len(retrieved_chunks))

        prompt = get_prompt(query, retrieved_chunks)

        try:
            answer = self.llm.complete(prompt, model=LLM_MODEL)
        except Exception:
            logger.exception("Failed to retrieve answer from LLM")
            return None

        return answer
