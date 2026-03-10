"use client";

import type { PlanetResults } from "@/types";
import ResultCard from "./ResultCard";

interface Props {
  results: PlanetResults;
}

export default function ResultsPanel({ results }: Props) {
  return (
    <div className="py-12">
      <h3 className="font-display text-2xl font-semibold text-heading">
        Analysis Results
      </h3>
      <p className="mt-2 text-sm text-muted">
        Three models, three independent verdicts.
      </p>
      <div className="mt-8 grid gap-5 lg:grid-cols-3">
        {results.results.map((r, i) => (
          <ResultCard key={r.modelName} result={r} index={i} />
        ))}
      </div>
    </div>
  );
}
