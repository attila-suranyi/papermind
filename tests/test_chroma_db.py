from app.store.chroma_db import _get_single_query_results


def _make_query_result(ids, documents, metadatas, distances):
    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def test_get_single_query_results_basic():
    """Should return one dict per result with correct fields."""
    qr = _make_query_result(
        ids=["id1", "id2"],
        documents=["doc1", "doc2"],
        metadatas=[{"k": "v1"}, {"k": "v2"}],
        distances=[0.1, 0.2],
    )
    results = _get_single_query_results(qr)
    assert len(results) == 2
    assert results[0] == {"id": "id1", "document": "doc1", "metadata": {"k": "v1"}, "distance": 0.1}
    assert results[1] == {"id": "id2", "document": "doc2", "metadata": {"k": "v2"}, "distance": 0.2}


def test_get_single_query_results_empty():
    """Should return an empty list when there are no results."""
    qr = _make_query_result(ids=[], documents=[], metadatas=[], distances=[])
    results = _get_single_query_results(qr)
    assert results == []


def test_get_single_query_results_missing_keys():
    """Should handle a QueryResult with missing optional keys gracefully."""
    results = _get_single_query_results({})
    assert results == []


def test_get_single_query_results_partial_fields():
    """Should fill None for missing document/metadata/distance entries."""
    qr = {
        "ids": [["id1"]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    results = _get_single_query_results(qr)
    assert len(results) == 1
    assert results[0]["id"] == "id1"
    assert results[0]["document"] is None
    assert results[0]["metadata"] is None
    assert results[0]["distance"] is None
