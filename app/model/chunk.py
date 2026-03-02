from typing import Optional, List


class Chunk:

    def __init__(self, text: str, metadata: dict, embedding: Optional[List[float]] = None):
        self.text = text
        self.metadata = metadata
        self.embedding = embedding