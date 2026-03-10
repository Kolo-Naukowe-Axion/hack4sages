"use client";

import type { ModelResult } from "@/types";

interface Props {
  result: ModelResult;
  index: number;
}

const verdictConfig = {
  detected: { label: "Biosignature Detected", color: "text-green", bg: "bg-green/8" },
  none: { label: "No Biosignature", color: "text-red", bg: "bg-red/8" },
  uncertain: { label: "Inconclusive", color: "text-amber", bg: "bg-amber/8" },
};

export default function ResultCard({ result, index }: Props) {
  const v = verdictConfig[result.verdict];
  const isQuantum = result.modelType === "quantum";

  return (
    <div
      className="border border-border bg-paper p-5"
      style={{ animation: `fadeSlideUp 0.4s ease-out ${index * 120}ms both` }}
    >
      <div className="flex items-baseline justify-between">
        <h4 className="font-serif text-base font-semibold text-heading">
          {result.modelName}
        </h4>
        <span className="font-sans text-xs text-muted">
          {isQuantum ? "Quantum" : "Classical"}
        </span>
      </div>

      <hr className="journal-rule my-3" />

      <div className={`inline-block px-2.5 py-1 ${v.bg}`}>
        <span className={`font-serif text-sm font-semibold ${v.color}`}>
          {v.label}
        </span>
      </div>

      <div className="mt-4 flex items-baseline gap-1">
        <span className="font-mono text-3xl font-bold text-heading">
          {result.confidence.toFixed(1)}
        </span>
        <span className="font-mono text-sm text-muted">%</span>
        <span className="ml-2 font-sans text-xs text-muted">confidence</span>
      </div>

      {result.detectedGases.length > 0 && (
        <div className="mt-4">
          <p className="mb-2 font-sans text-xs font-medium uppercase tracking-wider text-muted">
            Detected Species
          </p>
          <div className="space-y-1">
            {result.detectedGases.map((g) => (
              <div key={g.formula} className="flex items-baseline justify-between text-sm">
                <span className="text-text">
                  {g.name} <span className="font-mono text-xs text-muted">({g.formula})</span>
                </span>
                <span className="font-mono text-xs text-muted">{g.confidence.toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mt-4 border-t border-border-light pt-2 text-right">
        <span className="font-mono text-xs text-muted">
          t = {(result.processingTimeMs / 1000).toFixed(2)}s
        </span>
      </div>
    </div>
  );
}
