"use client";

import { Atom, Brain } from "lucide-react";
import type { ModelResult } from "@/types";

interface Props {
  result: ModelResult;
  index: number;
}

const verdictStyles = {
  detected: {
    label: "Biosignature Detected",
    color: "text-green",
    ring: "var(--color-green)",
    bg: "bg-green/5",
  },
  none: {
    label: "No Biosignature",
    color: "text-red",
    ring: "var(--color-red)",
    bg: "bg-red/5",
  },
  uncertain: {
    label: "Uncertain",
    color: "text-amber",
    ring: "var(--color-amber)",
    bg: "bg-amber/5",
  },
};

export default function ResultCard({ result, index }: Props) {
  const v = verdictStyles[result.verdict];
  const isQuantum = result.modelType === "quantum";
  const pct = result.confidence;

  return (
    <div
      className="flex flex-col rounded-2xl bg-deep p-6 shadow-sm transition-shadow duration-300 hover:shadow-md lg:p-8"
      style={{ animation: `fadeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) ${index * 120}ms both` }}
    >
      <div className="flex items-start justify-between">
        <h3 className="font-display text-lg font-semibold text-heading">
          {result.modelName}
        </h3>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
            isQuantum ? "bg-cyan/8 text-cyan" : "bg-surface text-muted"
          }`}
        >
          <span>{isQuantum ? <Atom size={11} /> : <Brain size={11} />}</span>
          {isQuantum ? "Quantum" : "Classical"}
        </span>
      </div>

      <div className="mt-8 flex items-center gap-6">
        <div
          className="relative flex h-20 w-20 shrink-0 items-center justify-center rounded-full"
          style={{
            background: `conic-gradient(${v.ring} ${pct}%, var(--color-border) ${pct}%)`,
          }}
        >
          <div className="flex h-[68px] w-[68px] items-center justify-center rounded-full bg-deep">
            <span className="font-mono text-lg font-bold text-heading">
              {pct.toFixed(0)}%
            </span>
          </div>
        </div>

        <div>
          <p className={`font-display text-base font-semibold ${v.color}`}>
            {v.label}
          </p>
          <p className="mt-0.5 text-xs text-muted">confidence score</p>
        </div>
      </div>

      {result.detectedGases.length > 0 && (
        <div className="mt-8">
          <p className="mb-2.5 text-xs font-medium uppercase tracking-wider text-muted">
            Detected Gases
          </p>
          <div className="flex flex-wrap gap-2">
            {result.detectedGases.map((g) => (
              <span
                key={g.formula}
                className="rounded-full border border-border bg-surface/60 px-3 py-1 text-sm text-heading transition-colors hover:border-heading/20"
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

      <div className="mt-auto pt-6 text-right">
        <span className="font-mono text-xs text-muted">
          {(result.processingTimeMs / 1000).toFixed(2)}s
        </span>
      </div>
    </div>
  );
}
