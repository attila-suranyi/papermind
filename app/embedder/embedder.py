from functorch.dim import Tensor
from sentence_transformers import SentenceTransformer

class Embedder:

    model: SentenceTransformer

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed(self, text: str) -> Tensor:
        embedding = Tensor
        try:
            embedding = self.model.encode(text, convert_to_numpy=False)
        except Exception as e:
            print(f"Failed to embed text: {e}")

        return embedding