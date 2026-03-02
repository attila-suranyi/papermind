from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    metadata: dict
    embedding: Optional[List[float]] = None