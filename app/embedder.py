from typing import List
from sentence_transformers import SentenceTransformer


class Embedder:

    model: SentenceTransformer

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()