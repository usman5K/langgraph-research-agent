from pathlib import Path

from agent.tools.docs_search import HashingEmbeddings, get_store, load_chunks

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_chunks_are_split_by_section():
    chunks = load_chunks(DOCS)
    assert len(chunks) >= 9
    assert {"source", "section"} <= chunks[0].metadata.keys()


def test_offline_retrieval_finds_the_right_doc(monkeypatch):
    monkeypatch.setattr("agent.tools.docs_search.make_embeddings", HashingEmbeddings)
    get_store.cache_clear()
    hits = get_store(str(DOCS)).similarity_search(
        "who gets paged when the primary does not acknowledge", k=1
    )
    assert hits[0].metadata["source"] == "oncall.md"
    get_store.cache_clear()
