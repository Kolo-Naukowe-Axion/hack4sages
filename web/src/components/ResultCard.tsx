"use client";

import { Atom, Brain } from "lucide-react";
import type { ModelResult } from "@/types";

interface Props {
  result: ModelResult;
  index: number;
}

const verdictStyles = {
  detected: {
    label: "BIOSIGNATURE DETECTED",
    color: "text-green",
    shadow: "0 0 20px rgba(16,185,129,0.3)",
    ring: "var(--color-green)",
  },
  none: {
    label: "NO BIOSIGNATURE",
    color: "text-red",
    shadow: "0 0 20px rgba(239,68,68,0.3)",
    ring: "var(--color-red)",
  },
  uncertain: {
    label: "UNCERTAIN",
    color: "text-amber",
    shadow: "0 0 20px rgba(245,158,11,0.3)",
    ring: "var(--color-amber)",
  },
};

export default function ResultCard({ result, index }: Props) {
  const v = verdictStyles[result.verdict];
  const isQuantum = result.modelType === "quantum";
  const pct = result.confidence;

  return (
    <div
      className="rounded-2xl border border-border bg-deep/60 p-6 backdrop-blur-md transition-all duration-300 hover:border-cyan/20 hover:shadow-lg hover:shadow-cyan/5"
      style={{ animation: `fadeSlideUp 0.5s ease-out ${index * 150}ms both` }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="font-display text-lg font-semibold text-heading">
          {result.modelName}
        </h3>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${
            isQuantum ? "bg-cyan/10 text-cyan" : "bg-muted/20 text-muted"
          }`}
        >
          {isQuantum ? <Atom size={12} /> : <Brain size={12} />}
          {isQuantum ? "Quantum" : "Classical"}
        </span>
      </div>

      {/* Verdict + Confidence Ring */}
      <div className="mt-6 flex items-center gap-6">
        {/* Confidence ring */}
        <div
          className="relative flex h-24 w-24 shrink-0 items-center justify-center rounded-full"
          style={{
            background: `conic-gradient(${v.ring} ${pct}%, var(--color-border) ${pct}%)`,
          }}
        >
          <div className="flex h-[84px] w-[84px] items-center justify-center rounded-full bg-deep">
            <span className="font-mono text-xl font-bold text-heading">
              {pct.toFixed(1)}%
            </span>
          </div>
        </div>

        <div>
          <p
            className={`font-display text-lg font-bold ${v.color}`}
            style={{ textShadow: v.shadow }}
          >
            {v.label}
          </p>
          <p className="mt-1 text-sm text-muted">
            confidence score
          </p>
        </div>
      </div>

      {/* Detected Gases */}
      {result.detectedGases.length > 0 && (
        <div className="mt-6">
          <p className="mb-2 text-xs uppercase tracking-wide text-muted">
            Detected Gases
          </p>
          <div className="flex flex-wrap gap-2">
            {result.detectedGases.map((g) => (
              <span
                key={g.formula}
                className="rounded-full border border-border bg-surface px-3 py-1 text-sm text-text transition-colors hover:border-cyan/30"
              >
                {g.formula}
                <span className="ml-1.5 font-mono text-xs text-muted">
                  {g.confidence.toFixed(0)}%
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Processing time */}
      <div className="mt-6 text-right">
        <span className="font-mono text-sm text-muted">
          {(result.processingTimeMs / 1000).toFixed(2)}s
        </span>
      </div>
    </div>
  );
}
