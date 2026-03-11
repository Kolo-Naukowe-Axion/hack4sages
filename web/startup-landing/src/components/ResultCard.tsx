import { ModelResult } from "@/types";
import { Cpu, TreePine } from "lucide-react";

interface Props {
  result: ModelResult;
}

const verdictStyles = {
  detected: { bg: "bg-green/10", text: "text-green", label: "Detected" },
  uncertain: { bg: "bg-amber/10", text: "text-amber", label: "Uncertain" },
  none: { bg: "bg-muted/10", text: "text-muted", label: "None" },
};

export function ResultCard({ result }: Props) {
  const v = verdictStyles[result.verdict];

  return (
    <div className="p-5 rounded-xl border border-border bg-surface/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {result.modelType === "quantum" ? (
            <Cpu className="w-4 h-4 text-purple" />
          ) : (
            <TreePine className="w-4 h-4 text-green" />
          )}
          <span className="text-sm font-semibold">{result.modelName}</span>
        </div>
        <span className="text-[10px] px-2 py-0.5 rounded-full border border-border text-muted font-mono">
          {result.modelType}
        </span>
      </div>

      <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg ${v.bg} mb-4`}>
        <span className={`text-sm font-bold ${v.text}`}>{v.label}</span>
        <span className={`text-xs font-mono ${v.text}`}>{result.confidence.toFixed(1)}%</span>
      </div>

      {result.detectedGases.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] text-muted uppercase tracking-wider mb-2">Gases</div>
          <div className="flex flex-wrap gap-1.5">
            {result.detectedGases.map((g) => (
              <span
                key={g.formula}
                className="text-xs px-2 py-1 rounded-md bg-bg border border-border font-mono"
              >
                {g.formula}{" "}
                <span className="text-muted">{g.confidence.toFixed(0)}%</span>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="text-[11px] text-muted font-mono">
        {result.processingTimeMs.toLocaleString()} ms
      </div>
    </div>
  );
}
