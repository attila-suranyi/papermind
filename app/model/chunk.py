from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Chunk:
    text: str
    metadata: dict
    embedding: Optional[List[List[float]]] = None