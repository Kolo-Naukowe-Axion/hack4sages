import { ModelResult } from "@/types";
import { Cpu, Zap } from "lucide-react";

interface ResultsOutputProps {
  results: [ModelResult, ModelResult, ModelResult];
}

const verdictStyles: Record<string, string> = {
  detected: "bg-green-100 text-green-800 border-green-300",
  uncertain: "bg-yellow-100 text-yellow-800 border-yellow-300",
  none: "bg-red-100 text-red-800 border-red-300",
};

const verdictLabels: Record<string, string> = {
  detected: "BIOSIGNATURE DETECTED",
  uncertain: "UNCERTAIN",
  none: "NO BIOSIGNATURE",
};

export default function ResultsOutput({ results }: ResultsOutputProps) {
  return (
    <div className="space-y-3">
      <div className="font-mono text-[12px] text-nb-muted mb-2">
        # Analysis complete. Results from 3 models:
      </div>
      <div className="grid gap-3">
        {results.map((r, i) => (
          <div key={i} className="border border-nb-border rounded bg-white p-3">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {r.modelType === "quantum" ? (
                  <Zap size={14} className="text-nb-blue" />
                ) : (
                  <Cpu size={14} className="text-nb-muted" />
                )}
                <span className="font-mono text-[13px] font-semibold">{r.modelName}</span>
                <span className="text-[11px] px-1.5 py-0.5 bg-nb-code-bg rounded font-mono text-nb-muted">
                  {r.modelType}
                </span>
              </div>
              <span className="font-mono text-[11px] text-nb-muted">
                {r.processingTimeMs}ms
              </span>
            </div>

            <div className="flex items-center gap-3 mb-2">
              <span className={`text-[11px] font-mono px-2 py-0.5 rounded border ${verdictStyles[r.verdict]}`}>
                {verdictLabels[r.verdict]}
              </span>
              <span className="font-mono text-[13px]">
                Confidence: <strong>{r.confidence.toFixed(1)}%</strong>
              </span>
            </div>

            {r.detectedGases.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-2">
                {r.detectedGases.map((g, j) => (
                  <span key={j} className="text-[11px] font-mono px-2 py-0.5 bg-nb-code-bg rounded border border-nb-border">
                    {g.formula} ({g.confidence.toFixed(1)}%)
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
