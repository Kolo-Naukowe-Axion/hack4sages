"use client";

import { ModelResult, Planet } from "@/types";
import { SpectrumChart } from "./SpectrumChart";

const verdictStyles: Record<string, { bg: string; text: string; label: string }> = {
  detected: { bg: "bg-accent-green/10", text: "text-accent-green", label: "Biosignature Detected" },
  uncertain: { bg: "bg-accent-amber/10", text: "text-accent-amber", label: "Uncertain" },
  none: { bg: "bg-surface", text: "text-muted", label: "None Detected" },
};

interface Props {
  results: ModelResult[];
  planet: Planet;
}

export function ResultsPanel({ results, planet }: Props) {
  return (
    <div>
      <h3 className="text-[11px] uppercase tracking-wider text-muted mb-4">
        Analysis Results
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 stagger-children">
        {results.map((r, i) => {
          const v = verdictStyles[r.verdict];
          return (
            <div
              key={i}
              className="bg-surface border border-border rounded-lg overflow-hidden hover:border-muted/30 transition-all duration-300"
            >
              <div className="p-4 border-b border-border">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-medium text-text text-sm">{r.modelName}</span>
                  <span
                    className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded-md ${
                      r.modelType === "quantum"
                        ? "text-accent-blue bg-accent-blue/10"
                        : "text-accent-green bg-accent-green/10"
                    }`}
                  >
                    {r.modelType}
                  </span>
                </div>

                <div className={`text-xs font-medium px-2 py-1 rounded-md inline-block mb-3 ${v.bg} ${v.text}`}>
                  {v.label}
                </div>

                <div className="flex items-center gap-4 text-sm">
                  <div>
                    <span className="text-muted text-[11px]">Confidence</span>
                    <div className="flex items-center gap-2 mt-0.5">
                      <div className="w-20 h-1.5 bg-border rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${r.confidence}%`,
                            background:
                              r.confidence > 75
                                ? "#22c55e"
                                : r.confidence > 50
                                ? "#f59e0b"
                                : "#ef4444",
                          }}
                        />
                      </div>
                      <span className="text-text font-mono text-xs">
                        {r.confidence.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div>
                    <span className="text-muted text-[11px]">Time</span>
                    <p className="text-text font-mono text-xs mt-0.5">
                      {r.processingTimeMs} ms
                    </p>
                  </div>
                </div>
              </div>

              {r.detectedGases.length > 0 && (
                <div className="p-4 border-b border-border">
                  <div className="text-[11px] text-muted mb-2">Detected Gases</div>
                  <div className="flex flex-wrap gap-1.5">
                    {r.detectedGases.map((g) => (
                      <span
                        key={g.formula}
                        className="bg-elevated border border-border rounded-md px-2 py-0.5 text-xs"
                      >
                        <span className="text-text font-mono">{g.formula}</span>
                        <span className="text-muted ml-1">{g.confidence.toFixed(0)}%</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {r.spectrumHighlights.length > 0 && (
                <div className="p-4">
                  <SpectrumChart
                    data={planet.spectrumData}
                    highlights={r.spectrumHighlights}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
