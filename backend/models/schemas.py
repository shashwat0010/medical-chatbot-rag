from typing import List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    title: str
    journal: str
    year: Optional[int] = None
    pubmed_url: str
    pmid: str
    authors: Optional[str] = None


class PaperMetadata(BaseModel):
    pmid: str
    title: str
    abstract: str
    journal: str
    year: Optional[int] = None
    authors: Optional[str] = None
    pubmed_url: str
    doi: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Medical research question")
    max_papers: Optional[int] = Field(None, ge=1, le=30)


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence_note: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    insufficient_evidence: bool = False
    sources_searched: List[str] = Field(default_factory=lambda: ["PubMed"])
    confidence_label: Optional[str] = None


class SearchPapersRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    max_results: Optional[int] = Field(None, ge=1, le=30)


class SearchPapersResponse(BaseModel):
    papers: List[PaperMetadata]
    total: int
    query: str


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_configured: bool
