"use client";

import ModelCard from "@/components/ModelCard";
import ComparisonTable from "@/components/ComparisonTable";
import Footer from "@/components/Footer";

const models = [
  {
    title: "QELM — Vetrano Architecture",
    modelType: "quantum" as const,
    description:
      "Reproduction of Vetrano et al. 2025 — the first quantum atmospheric retrieval on real hardware. Angle encoding into a random quantum reservoir, Z-basis measurement, linear output via SVD.",
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
      "Our modified circuit with different entanglement pattern and adjusted depth. Tests whether topology affects classification performance on biosignature data.",
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
      "The strongest traditional ML model — Random Forest with maximum hyperparameter optimization. The benchmark everything else is measured against.",
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
  return (
    <div className="mx-auto max-w-7xl px-6 py-12 lg:px-8">
      <h1 className="font-display text-4xl font-bold text-heading">Models</h1>
      <p className="mt-3 max-w-2xl text-lg text-muted">
        Three approaches to biosignature detection — quantum and classical,
        compared side by side.
      </p>

      <div className="mt-10 space-y-6">
        {models.map((m) => (
          <ModelCard key={m.title} {...m} />
        ))}
      </div>

      <ComparisonTable />
      <Footer />
    </div>
  );
}
