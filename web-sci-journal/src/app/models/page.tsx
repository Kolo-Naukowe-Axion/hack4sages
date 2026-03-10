"use client";

import ModelCard from "@/components/ModelCard";
import ComparisonTable from "@/components/ComparisonTable";
import Footer from "@/components/Footer";
import { useScrollReveal } from "@/hooks/useScrollReveal";

const models = [
  {
    title: "QELM — Vetrano Architecture",
    modelType: "quantum" as const,
    description:
      "Reproduction of Vetrano et al. (2025) — the first quantum atmospheric retrieval on real hardware. Input spectra are angle-encoded via RX gates into a random quantum reservoir with RY and CNOT entanglement, measured in the Z-basis, and classified through a linear output layer computed via SVD.",
    steps: [
      { label: "Spectrum", type: "classical" as const },
      { label: "PCA", type: "classical" as const },
      { label: "RX Encoding", type: "quantum" as const },
      { label: "RY+CNOT Reservoir", type: "quantum" as const },
      { label: "Z-Measurement", type: "quantum" as const },
      { label: "SVD", type: "classical" as const },
      { label: "Output", type: "classical" as const },
    ],
    stats: [
      { label: "Accuracy", value: "94.2%" },
      { label: "Precision", value: "92.8%" },
      { label: "Recall", value: "95.6%" },
      { label: "Hardware", value: "Odra 5 (IQM Spark)" },
      { label: "Qubits", value: "5" },
      { label: "Training Time", value: "~45 min" },
    ],
    defaultOpen: true,
  },
  {
    title: "QELM — Extended Topology",
    modelType: "quantum" as const,
    description:
      "Our modified circuit architecture with a different entanglement pattern and adjusted circuit depth. This variant tests whether reservoir topology significantly affects classification performance on biosignature spectral data.",
    steps: [
      { label: "Spectrum", type: "classical" as const },
      { label: "PCA", type: "classical" as const },
      { label: "Angle Encoding", type: "quantum" as const },
      { label: "Modified Reservoir", type: "quantum" as const },
      { label: "Measurement", type: "quantum" as const },
      { label: "SVD", type: "classical" as const },
      { label: "Output", type: "classical" as const },
    ],
    stats: [
      { label: "Accuracy", value: "91.7%" },
      { label: "Precision", value: "90.3%" },
      { label: "Recall", value: "93.1%" },
      { label: "Hardware", value: "VTT Q50" },
      { label: "Qubits", value: "53" },
      { label: "Training Time", value: "~30 min" },
    ],
    defaultOpen: false,
  },
  {
    title: "Classical Baseline — Random Forest",
    modelType: "classical" as const,
    description:
      "A hyperparameter-optimized Random Forest ensemble serving as the classical benchmark. Trained on identical PCA-reduced spectral features to enable direct comparison with quantum approaches.",
    steps: [
      { label: "Spectrum", type: "classical" as const },
      { label: "PCA", type: "classical" as const },
      { label: "Feature Engineering", type: "classical" as const },
      { label: "Ensemble/RF", type: "classical" as const },
      { label: "Output", type: "classical" as const },
    ],
    stats: [
      { label: "Accuracy", value: "96.8%" },
      { label: "Precision", value: "95.4%" },
      { label: "Recall", value: "98.1%" },
      { label: "Hardware", value: "GPU" },
      { label: "Qubits", value: "—" },
      { label: "Training Time", value: "~5 min" },
    ],
    defaultOpen: false,
  },
];

export default function ModelsPage() {
  const cardsRef = useScrollReveal();
  const tableRef = useScrollReveal();

  return (
    <article className="mx-auto max-w-4xl px-6 py-12 lg:py-16">
      <header>
        <p className="section-number text-sm">§5</p>
        <h1 className="font-serif text-3xl font-bold text-heading lg:text-4xl">
          Methods
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-muted">
          Three classification approaches are employed for biosignature detection —
          two quantum extreme learning machines and one classical baseline,
          enabling direct comparison across computational paradigms.
        </p>
      </header>

      <hr className="journal-rule mt-6" />

      <div ref={cardsRef} className="mt-8 space-y-5">
        {models.map((m, i) => (
          <div
            key={m.title}
            className="reveal opacity-0 translate-y-4 transition-all duration-700"
            style={{ transitionDelay: `${i * 80}ms` }}
          >
            <ModelCard {...m} />
          </div>
        ))}
      </div>

      <div ref={tableRef}>
        <div className="reveal opacity-0 translate-y-4 transition-all duration-700">
          <ComparisonTable />
        </div>
        <div className="reveal opacity-0 translate-y-4 transition-all duration-700">
          <Footer />
        </div>
      </div>
    </article>
  );
}
