"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";

const models = [
  {
    color: "bg-cyan",
    name: "QELM Vetrano",
    desc: "Quantum extreme learning machine based on Vetrano et al. 2025 — 5-qubit IQM Spark hardware.",
  },
  {
    color: "bg-teal",
    name: "QELM Extended",
    desc: "Extended quantum reservoir with optimized feature mapping for atmospheric retrieval.",
  },
  {
    color: "bg-muted",
    name: "Classical RF",
    desc: "Random forest baseline trained on identical spectral features for direct comparison.",
  },
];

export default function BridgePanel() {
  return (
    <div className="rounded-2xl bg-deep p-8 shadow-lg shadow-black/5 lg:p-10">
      <div className="grid gap-10 lg:grid-cols-[45fr_55fr]">
        <div>
          <h3 className="font-display text-2xl font-semibold text-heading">
            Under the Hood
          </h3>
          <p className="mt-3 leading-relaxed text-muted">
            Three models, three approaches, one question: is there life? Each
            model processes the same transmission spectrum independently —
            quantum and classical, side by side.
          </p>
        </div>

        <div className="space-y-4">
          {models.map((m) => (
            <div key={m.name} className="flex items-start gap-3">
              <span
                className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${m.color}`}
              />
              <div>
                <p className="font-display font-semibold text-heading">
                  {m.name}
                </p>
                <p className="text-sm text-muted">{m.desc}</p>
              </div>
            </div>
          ))}

          <div className="pt-2">
            <Link
              href="/models"
              className="inline-flex items-center gap-2 rounded-full border border-cyan/30 px-5 py-2.5 text-sm font-medium text-cyan transition-colors hover:bg-cyan/5"
            >
              Explore Models
              <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
