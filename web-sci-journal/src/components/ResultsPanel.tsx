"use client";

import type { PlanetResults } from "@/types";
import ResultCard from "./ResultCard";

interface Props {
  results: PlanetResults;
}

export default function ResultsPanel({ results }: Props) {
  return (
    <div className="py-10">
      <p className="section-number text-sm">§3.2</p>
      <h2 className="font-serif text-2xl font-semibold text-heading">
        Detection Results
      </h2>
      <p className="mt-2 max-w-2xl text-sm text-muted">
        Independent classification results from three models applied to the same transmission spectrum.
        Statistical confidence represents the posterior probability of the assigned verdict.
      </p>

      <div className="mt-6 grid gap-5 lg:grid-cols-3">
        {results.results.map((r, i) => (
          <ResultCard key={r.modelName} result={r} index={i} />
        ))}
      </div>
    </div>
  );
}
