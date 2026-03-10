"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";

const models = [
  {
    name: "QELM Vetrano",
    desc: "Quantum extreme learning machine — 5-qubit IQM Spark hardware.",
  },
  {
    name: "QELM Extended",
    desc: "Extended quantum reservoir with optimized feature mapping.",
  },
  {
    name: "Classical RF",
    desc: "Random forest baseline for direct comparison.",
  },
];

export default function BridgePanel() {
  return (
    <div className="rounded-2xl bg-deep p-8 shadow-sm lg:p-10">
      <div className="grid gap-10 lg:grid-cols-[45fr_55fr]">
        <div>
          <h3 className="font-display text-2xl font-semibold tracking-tight text-heading">
            Under the Hood
          </h3>
          <p className="mt-3 text-sm leading-relaxed text-muted">
            Three models process the same transmission spectrum independently —
            quantum and classical approaches, compared side by side.
          </p>
        </div>

        <div className="space-y-4">
          {models.map((m, i) => (
            <div key={m.name} className="flex items-start gap-3">
              <span
                className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${
                  i === 0 ? "bg-cyan" : i === 1 ? "bg-teal" : "bg-muted"
                }`}
              />
              <div>
                <p className="text-sm font-semibold text-heading">{m.name}</p>
                <p className="text-xs text-muted">{m.desc}</p>
              </div>
            </div>
          ))}

          <div className="pt-3">
            <Link
              href="/models"
              className="inline-flex items-center gap-2 rounded-full bg-surface px-5 py-2.5 text-sm font-medium text-heading transition-all hover:bg-border/60 focus:outline-none focus-visible:ring-2 focus-visible:ring-heading/20"
            >
              Explore Models
              <ArrowRight size={14} />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
