"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";

const models = [
  {
    name: "QELM Vetrano",
    desc: "Quantum extreme learning machine, Vetrano et al. (2025). 5-qubit IQM Spark hardware.",
  },
  {
    name: "QELM Extended",
    desc: "Modified quantum reservoir with optimized entanglement topology for atmospheric retrieval.",
  },
  {
    name: "Classical RF",
    desc: "Random forest baseline trained on identical spectral features for direct comparison.",
  },
];

export default function BridgePanel() {
  return (
    <div className="border-t border-border pt-8">
      <p className="section-number text-sm">§4</p>
      <h2 className="font-serif text-2xl font-semibold text-heading">
        Methodology Overview
      </h2>
      <p className="mt-2 max-w-2xl text-muted">
        Three models process the same transmission spectrum independently — two
        quantum, one classical — enabling direct comparison of detection capabilities.
      </p>

      <div className="mt-6 space-y-4">
        {models.map((m, i) => (
          <div key={m.name} className="flex items-baseline gap-3">
            <span className="font-mono text-xs text-muted">{i + 1}.</span>
            <div>
              <span className="font-serif font-semibold text-heading">{m.name}</span>
              <span className="mx-1.5 text-border">—</span>
              <span className="text-sm text-muted">{m.desc}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6">
        <Link
          href="/models"
          className="inline-flex items-center gap-2 border-b border-accent pb-0.5 font-serif text-sm font-medium text-accent transition-colors hover:text-accent-light"
        >
          Full methods & comparison
          <ArrowRight size={14} />
        </Link>
      </div>
    </div>
  );
}
