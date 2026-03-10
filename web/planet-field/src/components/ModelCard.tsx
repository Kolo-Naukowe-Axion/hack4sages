import { Zap, Cpu } from "lucide-react";

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
    <div className="bg-space-800/60 backdrop-blur-md border border-white/8 rounded-2xl overflow-hidden group hover:border-white/15 transition-all">
      <div className="p-6 border-b border-white/5">
        <div className="flex items-center gap-3 mb-3">
          <div
            className={`w-10 h-10 rounded-xl flex items-center justify-center ${
              isQuantum ? "bg-cyan/10" : "bg-amber/10"
            }`}
          >
            {isQuantum ? (
              <Zap className="w-5 h-5 text-cyan" />
            ) : (
              <Cpu className="w-5 h-5 text-amber" />
            )}
          </div>
          <div>
            <h3 className="font-bold text-white text-lg">{name}</h3>
            <span
              className={`text-[10px] uppercase tracking-wider font-medium ${
                isQuantum ? "text-cyan/60" : "text-amber/60"
              }`}
            >
              {type} model
            </span>
          </div>
        </div>
        <p className="text-sm text-white/50 leading-relaxed">{description}</p>
      </div>

      <div className="p-6 border-b border-white/5">
        <h4 className="text-[10px] uppercase tracking-wider text-white/30 mb-3">
          Architecture Pipeline
        </h4>
        <div className="space-y-2">
          {pipeline.map((step, i) => (
            <div key={i} className="flex items-start gap-3">
              <div className="flex flex-col items-center mt-1">
                <div
                  className={`w-2 h-2 rounded-full ${
                    isQuantum ? "bg-cyan" : "bg-amber"
                  }`}
                />
                {i < pipeline.length - 1 && (
                  <div className="w-px h-5 bg-white/10 mt-0.5" />
                )}
              </div>
              <div>
                <span className="text-sm text-white font-medium">{step.label}</span>
                <p className="text-xs text-white/40">{step.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-2 gap-3">
          {stats.map((s) => (
            <div key={s.label} className="bg-white/3 rounded-lg px-3 py-2">
              <div className="text-[10px] uppercase tracking-wide text-white/30 mb-0.5">
                {s.label}
              </div>
              <div
                className={`text-sm font-bold ${
                  isQuantum ? "text-cyan" : "text-amber"
                }`}
              >
                {s.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
