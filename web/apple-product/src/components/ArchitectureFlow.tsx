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
    <div className="flex flex-wrap items-center gap-1.5">
      {steps.map((step, i) => (
        <div key={step.label} className="flex items-center gap-1.5">
          {i > 0 && <ChevronRight size={12} className="shrink-0 text-border" />}
          <span
            className={`whitespace-nowrap rounded-full px-3.5 py-1.5 font-mono text-xs ${
              step.type === "quantum"
                ? "bg-cyan/8 text-cyan"
                : "bg-surface text-muted"
            }`}
          >
            {step.label}
          </span>
        </div>
      ))}
    </div>
  );
}
