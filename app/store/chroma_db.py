from pathlib import Path
from typing import Any, List

import chromadb
from chromadb import QueryResult

from app.model import Chunk
from app.retrieval.retrieved_chunk import RetrievedChunk


class ChromaDB:
    def __init__(self, db_path: Path, collection_name: str = "pdf_collection"):
        if not db_path.exists():
            db_path.mkdir(parents=True, exist_ok=True)

        if not db_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {str(db_path)}")

        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunk(self, chunk_id: str, chunk: Chunk):
        """Add a single chunk to the collection."""
        if chunk.embedding is None:
            raise ValueError(f"Chunk {chunk_id} has no embedding")

        self.collection.add(
            ids=[chunk_id],
            embeddings=[chunk.embedding],
            documents=[chunk.text],
            metadatas=[chunk.metadata],
        )

    def query_chunks(
        self, query_embedding: List[float], n_results: int = 5
    ) -> List[RetrievedChunk]:
        """Query the collection for similar chunks."""
        query_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        results = _get_single_query_results(query_result)
        return [RetrievedChunk.from_query_result(res) for res in results]


def _get_single_query_results(query_result: QueryResult) -> List[dict[str, Any]]:
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
