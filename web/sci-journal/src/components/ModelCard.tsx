"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
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
    <div className="border border-border bg-paper">
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center justify-between p-5 text-left lg:p-6"
      >
        <div className="flex items-baseline gap-3">
          <h3 className="font-serif text-lg font-semibold text-heading">
            {title}
          </h3>
          <span className="font-sans text-xs text-muted">
            {isQuantum ? "Quantum" : "Classical"}
          </span>
        </div>
        <span className="shrink-0 text-muted">
          {open ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </span>
      </button>

      {open && (
        <div className="border-t border-border-light px-5 pb-6 pt-5 lg:px-6">
          <p className="max-w-3xl text-sm leading-relaxed text-muted">{description}</p>

          <div className="mt-5">
            <p className="mb-2.5 font-sans text-xs font-medium uppercase tracking-wider text-muted">
              Pipeline
            </p>
            <ArchitectureFlow steps={steps} />
          </div>

          <div className="mt-6">
            <p className="mb-2.5 font-sans text-xs font-medium uppercase tracking-wider text-muted">
              Performance Metrics
            </p>
            <div className="grid gap-x-8 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
              {stats.map((s) => (
                <div
                  key={s.label}
                  className="flex items-baseline justify-between border-b border-border-light py-1.5"
                >
                  <span className="font-sans text-sm text-muted">{s.label}</span>
                  <span className="font-mono text-sm text-text">{s.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
