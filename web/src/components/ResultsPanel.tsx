"use client";

import type { PlanetResults } from "@/types";
import ResultCard from "./ResultCard";

interface Props {
  results: PlanetResults;
}

export default function ResultsPanel({ results }: Props) {
  return (
    <div className="min-h-[60vh] py-12">
      <div className="grid gap-6 lg:grid-cols-3">
        {results.results.map((r, i) => (
          <ResultCard key={r.modelName} result={r} index={i} />
        ))}
      </div>
    </div>
  );
}
