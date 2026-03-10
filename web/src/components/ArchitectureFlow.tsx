"use client";

import { ChevronRight } from "lucide-react";

interface Step {
  label: string;
  type: "quantum" | "classical";
}

interface Props {
  steps: Step[];
}

export default function ArchitectureFlow({ steps }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {steps.map((step, i) => (
        <div key={step.label} className="flex items-center gap-2">
          {i > 0 && <ChevronRight size={14} className="shrink-0 text-border" />}
          <span
            className={`whitespace-nowrap rounded-full border px-4 py-2 font-mono text-sm ${
              step.type === "quantum"
                ? "border-cyan/30 bg-cyan/10 text-cyan"
                : "border-border bg-surface text-muted"
            }`}
          >
            {step.label}
          </span>
        </div>
      ))}
    </div>
  );
}
