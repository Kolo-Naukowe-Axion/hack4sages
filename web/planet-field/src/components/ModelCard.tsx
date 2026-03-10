interface PipelineStep {
  label: string;
  detail: string;
}

interface Stat {
  label: string;
  value: string;
}

interface Props {
  name: string;
  type: "quantum" | "classical";
  description: string;
  pipeline: PipelineStep[];
  stats: Stat[];
}

export function ModelCard({ name, type, description, pipeline, stats }: Props) {
  const isQuantum = type === "quantum";

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden hover:border-muted/30 hover:-translate-y-1 transition-all duration-300">
      <div className="p-5 border-b border-border">
        <div className="flex items-center gap-3 mb-2">
          <h3 className="font-semibold text-text">{name}</h3>
          <span
            className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded-md ${
              isQuantum
                ? "text-accent-blue bg-accent-blue/10"
                : "text-accent-green bg-accent-green/10"
            }`}
          >
            {type}
          </span>
        </div>
        <p className="text-sm text-muted leading-relaxed">{description}</p>
      </div>

      <div className="p-5 border-b border-border">
        <h4 className="text-[11px] uppercase tracking-wider text-muted mb-3">
          Pipeline
        </h4>
        <div className="space-y-2">
          {pipeline.map((step, i) => (
            <div key={i} className="flex items-start gap-3">
              <div className="flex flex-col items-center mt-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-muted" />
                {i < pipeline.length - 1 && (
                  <div className="w-px h-5 bg-border mt-0.5" />
                )}
              </div>
              <div>
                <span className="text-sm text-text">{step.label}</span>
                <p className="text-xs text-muted">{step.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="p-5">
        <div className="grid grid-cols-2 gap-3">
          {stats.map((s) => (
            <div key={s.label} className="bg-elevated rounded-lg px-3 py-2">
              <div className="text-[11px] text-muted mb-0.5">{s.label}</div>
              <div className="text-sm font-mono font-medium text-text">
                {s.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
