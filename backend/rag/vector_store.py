import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import faiss
import numpy as np

from models.schemas import PaperMetadata
from services.embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    paper: PaperMetadata
    text: str
    score: float
    chunk_index: int


def _paper_to_chunks(paper: PaperMetadata) -> List[str]:
    header = f"Title: {paper.title}\nJournal: {paper.journal}\nYear: {paper.year or 'N/A'}\n"
    body = paper.abstract or ""
    full = header + body
    # Split long abstracts into ~800 char chunks with overlap
    max_len = 800
    overlap = 100
    if len(full) <= max_len:
        return [full]
    chunks: List[str] = []
    start = 0
    while start < len(full):
        end = min(start + max_len, len(full))
        chunks.append(full[start:end])
        if end >= len(full):
            break
        start = end - overlap
    return chunks


class FAISSVectorStore:
    def __init__(self) -> None:
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: List[RetrievedChunk] = []
        self._dimension: int = 0

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    async def build_from_papers(self, papers: List[PaperMetadata]) -> None:
        self._chunks = []
        texts: List[str] = []
        for paper in papers:
            for i, chunk_text in enumerate(_paper_to_chunks(paper)):
                self._chunks.append(
                    RetrievedChunk(paper=paper, text=chunk_text, score=0.0, chunk_index=i)
                )
                texts.append(chunk_text)

        if not texts:
            self._index = None
            return

        embeddings = await embed_texts(texts)
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self._dimension = vectors.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(vectors)
        logger.info("Built FAISS index with %d chunks from %d papers", len(texts), len(papers))

    async def search(self, query: str, top_k: int = 8) -> List[RetrievedChunk]:
        if not self.is_ready or self._index is None:
            return []

        query_vecs = await embed_query(query)
        q = np.array(query_vecs, dtype=np.float32)
        faiss.normalize_L2(q)
        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(q, k)

        results: List[RetrievedChunk] = []
        seen_pmids = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            if chunk.paper.pmid in seen_pmids and len(results) >= top_k:
                continue
            seen_pmids.add(chunk.paper.pmid)
            results.append(
                RetrievedChunk(
                    paper=chunk.paper,
                    text=chunk.text,
                    score=float(score),
                    chunk_index=chunk.chunk_index,
                )
            )
        return sorted(results, key=lambda c: c.score, reverse=True)
