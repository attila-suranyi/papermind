from app.retrieval.prompt import get_prompt
from app.retrieval.retrieved_chunk import RetrievedChunk


def _make_chunk(text="Some content.", filename="doc.pdf", pages=None):
    return RetrievedChunk(text=text, filename=filename, pages=pages or [1])


def test_get_prompt_returns_prompt_object():
    """get_prompt should return a Prompt with non-empty system and user prompts."""
    prompt = get_prompt("What is X?", [_make_chunk()])
    assert prompt.system_prompt
    assert prompt.user_prompt


def test_get_prompt_system_prompt_contains_instruction():
    """System prompt should instruct the assistant to use only the provided context."""
    prompt = get_prompt("Any question?", [])
    assert "context" in prompt.system_prompt.lower()
    assert "assistant" in prompt.system_prompt.lower()


def test_get_prompt_system_prompt_contains_citation_instruction():
    """System prompt should instruct the assistant to cite sources."""
    prompt = get_prompt("Any question?", [])
    assert "cite" in prompt.system_prompt.lower()


def test_get_prompt_user_prompt_contains_query():
    """User prompt should embed the original query."""
    query = "What is the meaning of life?"
    prompt = get_prompt(query, [])
    assert query in prompt.user_prompt


def test_get_prompt_user_prompt_contains_chunk_text():
    """User prompt should include the text of each retrieved chunk."""
    chunk = _make_chunk(text="Important finding.")
    prompt = get_prompt("Question?", [chunk])
    assert "Important finding." in prompt.user_prompt


def test_get_prompt_user_prompt_contains_source_reference():
    """User prompt should include the filename and page of each chunk."""
    chunk = _make_chunk(filename="paper.pdf", pages=[3, 4])
    prompt = get_prompt("Question?", [chunk])
    assert "paper.pdf" in prompt.user_prompt
    assert "3" in prompt.user_prompt


def test_get_prompt_handles_empty_context():
    """get_prompt should work without raising when context list is empty."""
    prompt = get_prompt("Question?", [])
    assert prompt.system_prompt
    assert "Question?" in prompt.user_prompt
