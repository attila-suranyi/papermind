from app.model import Prompt
from app.retrieval.retrieved_chunk import RetrievedChunk


def get_prompt(query: str, context: list[RetrievedChunk]) -> Prompt:
    context_text = ""
    for chunk in context:
        unit = f"""
        [Source: {chunk.filename}, pages: {chunk.pages}]
        {chunk.text}
        """
        context_text += unit

    system_prompt = """You are a helpful assistant. Answer the user's question based
    only on the provided context. If the answer cannot be found in the context, say so.
    After each claim in your answer, cite the source in parentheses,
    e.g. (introduction.pdf, page 3)."""

    user_prompt = f"""Context:
    {context_text}

    Question: {query}

    Answer:"""

    return Prompt(system_prompt=system_prompt, user_prompt=user_prompt)
