import json
from dataclasses import dataclass
from typing import List


@dataclass
class RetrievedChunk:
    text: str
    filename: str
    pages: List[int]

    @classmethod
    def from_query_result(cls, result: dict) -> "RetrievedChunk":
        try:
            origin_str = result["metadata"]["origin"]
            origin = json.loads(origin_str)
            filename = origin["filename"]
        except (json.JSONDecodeError, KeyError):
            filename = "unknown"

        try:
            doc_items_str = result["metadata"]["doc_items"]
            doc_items = json.loads(doc_items_str)
            pages = list({prov["page_no"] for item in doc_items for prov in item["prov"]})
        except (json.JSONDecodeError, KeyError):
            pages = []

        return cls(text=result["document"], filename=filename, pages=pages)
