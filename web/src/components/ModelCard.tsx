"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, Atom, Brain } from "lucide-react";
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
  defaultOpen?: boolean;
}

export default function ModelCard({
  title,
  modelType,
  description,
  steps,
  stats,
  defaultOpen = false,
}: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const isQuantum = modelType === "quantum";

  return (
    <div className="rounded-2xl bg-deep shadow-lg shadow-black/5 transition-all duration-300 hover:shadow-xl hover:shadow-black/10">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between p-6 text-left focus:outline-none lg:p-8"
      >
        <div className="flex items-center gap-4">
          <h3 className="font-display text-xl font-semibold text-heading">
            {title}
          </h3>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${
              isQuantum ? "bg-cyan/10 text-cyan" : "bg-surface text-muted"
            }`}
          >
            {isQuantum ? <Atom size={12} /> : <Brain size={12} />}
            {isQuantum ? "Quantum" : "Classical"}
          </span>
        </div>
        <span className="shrink-0 text-muted">
          {open ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </span>
      </button>

      {open && (
        <div className="border-t border-border px-6 pb-8 pt-6 lg:px-8">
          <p className="max-w-3xl leading-relaxed text-muted">{description}</p>

          <div className="mt-6">
            <p className="mb-3 text-xs uppercase tracking-wide text-muted">
              Pipeline
            </p>
            <ArchitectureFlow steps={steps} />
          </div>

          <div className="mt-8 grid gap-x-12 gap-y-2 sm:grid-cols-2 lg:grid-cols-3">
            {stats.map((s) => (
              <div
                key={s.label}
                className="flex items-baseline justify-between border-b border-border/50 py-2"
              >
                <span className="text-sm text-muted">{s.label}</span>
                <span className="font-mono text-sm text-cyan">{s.value}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
