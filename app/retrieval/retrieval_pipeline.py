import logging
from typing import Optional

from app.embedder import Embedder
from app.llm import LLMClient
from app.retrieval.prompt import get_prompt
from app.store.chroma_db import ChromaDB

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(
        self, db: ChromaDB, embedder: Embedder, llm: LLMClient, llm_model: Optional[str] = None
    ):
        self.embedder = embedder
        self.db = db
        self.llm = llm
        self.llm_model = llm_model

    def get_answer(self, query: str):
        query_embedding = self.embedder.embed(query)

        try:
            retrieved_chunks = self.db.query_chunks(query_embedding, n_results=5)
        except Exception:
            logger.exception("Failed to query collection")
            return None

        logger.info("Retrieved %d results for query", len(retrieved_chunks))

        prompt = get_prompt(query, retrieved_chunks)

        try:
            answer = self.llm.complete(prompt, model=self.llm_model)
        except Exception:
            logger.exception("Failed to retrieve answer from LLM")
            return None

        return answer
