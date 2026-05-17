import logging
from typing import List, Optional

from groq import RateLimitError as GroqRateLimitError
from openai import RateLimitError as OpenAIRateLimitError

from app.config import get_settings
from models.schemas import Citation, QueryResponse
from rag.extractive import build_extractive_answer
from rag.llm import MedicalLLM
from rag.scoring import compute_retrieval_confidence, retrieval_is_sufficient
from rag.vector_store import FAISSVectorStore, RetrievedChunk
from services.embeddings import EmbeddingServiceError, OpenAIQuotaError
from rag.formatting import paragraph_to_bullets
from services.guardrails import validate_answer_grounding, is_greeting_or_meta
from services.pubmed import prioritize_trusted_journals, search_pubmed

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self) -> None:
        self._vector_store = FAISSVectorStore()
        self._llm: Optional[MedicalLLM] = None

    def _get_llm(self) -> MedicalLLM:
        if self._llm is None:
            self._llm = MedicalLLM()
        return self._llm

    async def run(self, query: str, max_papers: Optional[int] = None) -> QueryResponse:
        settings = get_settings()

        # Handle greetings/meta queries early to skip PubMed search
        if is_greeting_or_meta(query):
            answer, cited_indices, _ = await self._get_llm().generate(query, [])
            return QueryResponse(
                answer=answer,
                citations=[],
                confidence_note="Assistant information.",
                confidence_score=1.0,
                insufficient_evidence=False,
                sources_searched=[],
            )

        use_local = settings.use_local_embeddings

        papers = await search_pubmed(query, max_results=max_papers)
        papers = prioritize_trusted_journals(papers)

        if not papers:
            return QueryResponse(
                answer="Current evidence is insufficient to provide a reliable answer.",
                citations=[],
                confidence_note=(
                    "No relevant papers were retrieved from PubMed for this query. "
                    "Try broader or alternative medical terminology."
                ),
                confidence_score=0.0,
                insufficient_evidence=True,
                sources_searched=["PubMed"],
            )

        try:
            await self._vector_store.build_from_papers(papers)
            chunks = await self._vector_store.search(
                query, top_k=settings.pubmed_retrieval_top_k
            )
        except (OpenAIQuotaError, EmbeddingServiceError) as exc:
            logger.error("Embedding failed: %s", exc)
            raise

        # Always use the best-matching chunks (do not over-filter TF-IDF scores)
        evidence_chunks = chunks[: max(settings.min_evidence_chunks, 5)]
        confidence_score = compute_retrieval_confidence(
            chunks, papers, use_local=use_local
        )

        if not retrieval_is_sufficient(chunks, papers, use_local=use_local):
            logger.info(
                "Weak retrieval: top_score=%.3f papers=%d",
                max((c.score for c in chunks), default=0),
                len(papers),
            )
            return QueryResponse(
                answer="Current evidence is insufficient to provide a reliable answer.",
                citations=self._chunks_to_citations(evidence_chunks[:3]),
                confidence_note=(
                    "Retrieved literature had very low semantic match to your question. "
                    "Try rephrasing with medical subject terms (drug class, condition, outcome)."
                ),
                confidence_score=confidence_score,
                insufficient_evidence=True,
                sources_searched=["PubMed"],
            )

        try:
            answer, cited_indices, llm_insufficient = await self._get_llm().generate(
                query, evidence_chunks
            )
        except (OpenAIRateLimitError, GroqRateLimitError) as exc:
            if "quota" in str(exc).lower() or "insufficient_quota" in str(exc).lower():
                logger.warning("LLM quota exceeded; using extractive fallback")
                answer, cited_indices = build_extractive_answer(query, evidence_chunks)
                llm_insufficient = False
            else:
                raise

        cited_chunks = self._resolve_citations(evidence_chunks, cited_indices)
        if not cited_chunks:
            cited_chunks = evidence_chunks[:3]

        citations = self._chunks_to_citations(cited_chunks)

        if not llm_insufficient and "\n- " not in answer and not answer.startswith("- "):
            answer = paragraph_to_bullets(answer)

        final_answer, insufficient, confidence_note = validate_answer_grounding(
            answer,
            llm_insufficient,
            confidence_score,
        )

        return QueryResponse(
            answer=final_answer,
            citations=citations,
            confidence_note=confidence_note,
            confidence_score=confidence_score,
            insufficient_evidence=insufficient,
            sources_searched=["PubMed (BMJ, Nature, Lancet, and peer-reviewed literature)"],
        )

    def _resolve_citations(
        self, chunks: List[RetrievedChunk], indices: List[int]
    ) -> List[RetrievedChunk]:
        result: List[RetrievedChunk] = []
        seen = set()
        for idx in indices:
            if 1 <= idx <= len(chunks):
                chunk = chunks[idx - 1]
                if chunk.paper.pmid not in seen:
                    seen.add(chunk.paper.pmid)
                    result.append(chunk)
        return result

    @staticmethod
    def _chunks_to_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
        citations: List[Citation] = []
        seen = set()
        for chunk in chunks:
            p = chunk.paper
            if p.pmid in seen:
                continue
            seen.add(p.pmid)
            citations.append(
                Citation(
                    title=p.title,
                    journal=p.journal,
                    year=p.year,
                    pubmed_url=p.pubmed_url,
                    pmid=p.pmid,
                    authors=p.authors,
                )
            )
        return citations
