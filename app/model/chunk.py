from dataclasses import dataclass
from typing import Optional
from functorch.dim import Tensor


@dataclass
class Chunk:
    text: str
    metadata: dict
    embedding: Optional[Tensor] = None