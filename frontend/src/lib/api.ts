const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Citation {
  title: string;
  journal: string;
  year?: number | null;
  pubmed_url: string;
  pmid: string;
  authors?: string | null;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  confidence_note: string;
  confidence_score: number;
  insufficient_evidence: boolean;
  sources_searched: string[];
}

export interface HealthResponse {
  status: string;
  version: string;
  llm_configured: boolean;
}

export async function queryMedicalResearch(query: string): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = err.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join(", ")
          : `Request failed (${res.status})`;
    throw new Error(message || `Request failed (${res.status})`);
  }

  return res.json();
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("API unreachable");
  return res.json();
}
