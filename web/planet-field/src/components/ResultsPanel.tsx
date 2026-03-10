"use client";

import { ModelResult } from "@/types";
import { SpectrumChart } from "./SpectrumChart";
import { Planet } from "@/types";
import { Zap, Cpu, FlaskConical } from "lucide-react";

const verdictStyles: Record<string, { bg: string; text: string; label: string }> = {
  detected: { bg: "bg-green/15", text: "text-green", label: "Biosignature Detected" },
  uncertain: { bg: "bg-amber/15", text: "text-amber", label: "Uncertain" },
  none: { bg: "bg-white/8", text: "text-white/50", label: "None Detected" },
};

interface Props {
  results: ModelResult[];
  planet: Planet;
}

export function ResultsPanel({ results, planet }: Props) {
  return (
    <div className="space-y-4">
      <h3 className="text-xs uppercase tracking-wider text-white/40">
        Analysis Results
      </h3>

      {results.map((r, i) => {
        const v = verdictStyles[r.verdict];
        return (
          <div
            key={i}
            className="bg-space-800/60 backdrop-blur-md border border-white/8 rounded-2xl overflow-hidden"
          >
            <div className="p-4 border-b border-white/5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  {r.modelType === "quantum" ? (
                    <Zap className="w-4 h-4 text-cyan" />
                  ) : (
                    <Cpu className="w-4 h-4 text-amber" />
                  )}
                  <span className="font-semibold text-white text-sm">{r.modelName}</span>
                  <span className="text-[10px] uppercase tracking-wide text-white/30 bg-white/5 px-2 py-0.5 rounded-full">
                    {r.modelType}
                  </span>
                </div>
                <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${v.bg} ${v.text}`}>
                  {v.label}
                </span>
              </div>

              <div className="flex items-center gap-6 text-sm">
                <div>
                  <span className="text-white/40 text-xs">Confidence</span>
                  <div className="flex items-center gap-2 mt-0.5">
                    <div className="w-24 h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${r.confidence}%`,
                          background:
                            r.confidence > 75
                              ? "#00ff88"
                              : r.confidence > 50
                              ? "#ffaa00"
                              : "#ff6644",
                        }}
                      />
                    </div>
                    <span className="text-white font-semibold text-xs">
                      {r.confidence.toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div>
                  <span className="text-white/40 text-xs">Time</span>
                  <p className="text-white text-xs font-medium mt-0.5">
                    {r.processingTimeMs} ms
                  </p>
                </div>
              </div>
            </div>

            {r.detectedGases.length > 0 && (
              <div className="p-4 border-b border-white/5">
                <div className="flex items-center gap-1.5 text-white/40 text-[10px] uppercase tracking-wide mb-2">
                  <FlaskConical className="w-3 h-3" />
                  Detected Gases
                </div>
                <div className="flex flex-wrap gap-2">
                  {r.detectedGases.map((g) => (
                    <span
                      key={g.formula}
                      className="bg-white/5 border border-white/8 rounded-lg px-2.5 py-1 text-xs"
                    >
                      <span className="text-white font-medium">{g.formula}</span>
                      <span className="text-white/40 ml-1.5">{g.confidence.toFixed(0)}%</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {r.spectrumHighlights.length > 0 && (
              <div className="p-4">
                <SpectrumChart data={planet.spectrumData} highlights={r.spectrumHighlights} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
