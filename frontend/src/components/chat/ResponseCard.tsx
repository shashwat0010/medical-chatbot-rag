"use client";

import { ExternalLink, FileText, ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StructuredAnswer } from "@/components/chat/StructuredAnswer";
import type { Citation, QueryResponse } from "@/lib/api";

interface ResponseCardProps {
  query: string;
  response: QueryResponse;
}

function confidenceVariant(score: number): "success" | "warning" | "secondary" {
  if (score >= 0.72) return "success";
  if (score >= 0.55) return "warning";
  return "secondary";
}

function CitationItem({ citation, index }: { citation: Citation; index: number }) {
  return (
    <li className="rounded-lg border border-border/60 bg-muted/30 p-3 text-sm">
      <div className="mb-1 flex items-start gap-2">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
          {index}
        </span>
        <div className="min-w-0 flex-1">
          <p className="font-medium leading-snug text-foreground">{citation.title}</p>
          <p className="mt-1 text-muted-foreground">
            {citation.journal}
            {citation.year ? ` · ${citation.year}` : ""}
            {citation.authors ? ` · ${citation.authors}` : ""}
          </p>
        </div>
      </div>
      <a
        href={citation.pubmed_url}
        target="_blank"
        rel="noopener noreferrer"
        className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
      >
        View on PubMed (PMID: {citation.pmid})
        <ExternalLink className="h-3 w-3" />
      </a>
    </li>
  );
}

export function ResponseCard({ query, response }: ResponseCardProps) {
  const scorePercent = Math.round(response.confidence_score * 100);

  return (
    <Card className="border-primary/10 shadow-md">
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <FileText className="h-4 w-4 text-primary" />
            Evidence-based answer
          </CardTitle>
          <Badge variant={confidenceVariant(response.confidence_score)}>
            Confidence: {scorePercent}%
          </Badge>
          {response.insufficient_evidence && (
            <Badge variant="warning">Insufficient evidence</Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground">Query: {query}</p>
      </CardHeader>
      <CardContent className="space-y-4">
        <StructuredAnswer text={response.answer} />

        <div className="flex items-start gap-2 rounded-lg border border-amber-200/80 bg-amber-50/80 p-3 text-sm text-amber-950 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-100">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p className="font-medium">Uncertainty note</p>
            <p className="mt-1 opacity-90">{response.confidence_note}</p>
          </div>
        </div>

        {response.citations.length > 0 && (
          <div>
            <h4 className="mb-2 text-sm font-semibold">Citations</h4>
            <ul className="space-y-2">
              {response.citations.map((c, i) => (
                <CitationItem key={c.pmid} citation={c} index={i + 1} />
              ))}
            </ul>
          </div>
        )}

        {response.sources_searched.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Sources: {response.sources_searched.join(", ")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
