"""search_docs: retrieval-augmented lookup over the markdown files in DOCS_DIR.

Docs are split by heading, embedded and kept in an in-memory vector store. With an
OpenAI key we use OpenAI embeddings; otherwise a small hashing embedder keeps the
tool (and the tests) fully offline. Swap InMemoryVectorStore for PGVector in prod.
"""

import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

from agent.config import get_settings

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings via the hashing trick (offline, no model)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def make_embeddings() -> Embeddings:
    settings = get_settings()
    use_openai = settings.embeddings == "openai" or (
        settings.embeddings == "auto" and settings.openai_api_key is not None
    )
    if use_openai:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key)
    return HashingEmbeddings()


def load_chunks(docs_dir: Path) -> list[Document]:
    """Split every markdown file into one chunk per `##` section."""
    chunks: list[Document] = []
    for path in sorted(docs_dir.glob("*.md")):
        sections = re.split(r"\n(?=## )", path.read_text(encoding="utf-8"))
        for section in sections:
            heading = section.strip().splitlines()[0].lstrip("# ").strip()
            chunks.append(
                Document(
                    page_content=section.strip(), metadata={"source": path.name, "section": heading}
                )
            )
    return chunks


@lru_cache(maxsize=4)
def get_store(docs_dir: str) -> InMemoryVectorStore:
    store = InMemoryVectorStore(make_embeddings())
    store.add_documents(load_chunks(Path(docs_dir)))
    return store


@tool
def search_docs(query: str) -> str:
    """Search the internal engineering handbook of Kestrel Analytics (on-call, deploys,
    architecture, SLOs). Use for any question about Kestrel's internal practices.
    Returns passages labelled with their source file and section; cite them.
    """
    hits = get_store(get_settings().docs_dir).similarity_search(query, k=3)
    if not hits:
        return "No matching documents."
    return "\n\n".join(
        f"[{d.metadata['source']} > {d.metadata['section']}]\n{d.page_content[:600]}" for d in hits
    )
