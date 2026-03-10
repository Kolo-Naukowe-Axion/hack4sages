"use client";

import { Atom, Brain } from "lucide-react";
import ArchitectureFlow from "./ArchitectureFlow";

interface Step {
  label: string;
  type: "quantum" | "classical";
}

interface Stat {
  label: string;
  value: string;
}

interface Props {
  title: string;
  modelType: "quantum" | "classical";
  description: string;
  steps: Step[];
  stats: Stat[];
}

export default function ModelCard({
  title,
  modelType,
  description,
  steps,
  stats,
}: Props) {
  const isQuantum = modelType === "quantum";

  return (
    <div className="flex w-[380px] shrink-0 flex-col rounded-2xl bg-deep p-7 shadow-sm transition-shadow duration-300 hover:shadow-md lg:w-[400px]">
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-display text-lg font-semibold leading-snug text-heading">
          {title}
        </h3>
        <span
          className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium ${
            isQuantum ? "bg-cyan/8 text-cyan" : "bg-surface text-muted"
          }`}
        >
          <span>{isQuantum ? <Atom size={11} /> : <Brain size={11} />}</span>
          {isQuantum ? "Quantum" : "Classical"}
        </span>
      </div>

      <p className="mt-3 text-sm leading-relaxed text-muted">{description}</p>

      <div className="mt-6">
        <p className="mb-2.5 text-[10px] font-medium uppercase tracking-widest text-muted">
          Pipeline
        </p>
        <ArchitectureFlow steps={steps} />
      </div>

      <div className="mt-6 grid grid-cols-2 gap-x-6">
        {stats.map((s) => (
          <div
            key={s.label}
            className="flex items-baseline justify-between border-b border-border/40 py-2"
          >
            <span className="text-xs text-muted">{s.label}</span>
            <span className="font-mono text-xs font-medium text-heading">{s.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
