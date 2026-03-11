"use client";

import { useScrollReveal } from "@/hooks/useScrollReveal";
import { Footer } from "@/components/Footer";
import { Cpu, TreePine } from "lucide-react";

interface ModelInfo {
  name: string;
  type: "quantum" | "classical";
  description: string;
  pipeline: string[];
  stats: { label: string; value: string }[];
}

const models: ModelInfo[] = [
  {
    name: "QELM Vetrano",
    type: "quantum",
    description:
      "Quantum extreme learning machine following the Vetrano et al. 2025 architecture. Encodes spectral features into a 5-qubit quantum reservoir via parameterized rotations and entangling gates. A single forward pass through the reservoir produces measurement statistics that are fed to a trained linear readout layer.",
    pipeline: [
      "PCA dimensionality reduction (50 → 5 features)",
      "Amplitude encoding into 5-qubit register",
      "Parameterized RY + CNOT reservoir layer",
      "Pauli-Z measurement statistics",
      "Linear SVM readout classifier",
    ],
    stats: [
      { label: "Qubits", value: "5" },
      { label: "Circuit Depth", value: "12" },
      { label: "Backend", value: "IQM Spark (Odra)" },
      { label: "Avg. Inference", value: "~2.3s" },
      { label: "Accuracy", value: "89.2%" },
      { label: "Training Set", value: "41k spectra" },
    ],
  },
  {
    name: "QELM Extended",
    type: "quantum",
    description:
      "Extended quantum ELM with deeper reservoir and additional measurement bases. Uses the same 5-qubit hardware but adds a second reservoir layer with different entanglement topology and measures in both Z and X bases, doubling the feature vector for the classical readout.",
    pipeline: [
      "PCA dimensionality reduction (50 → 5 features)",
      "Angle encoding with feature reuploading",
      "Two-layer reservoir (RY-RZ + ring CNOT)",
      "Pauli-Z and Pauli-X measurements",
      "Ridge regression readout classifier",
    ],
    stats: [
      { label: "Qubits", value: "5" },
      { label: "Circuit Depth", value: "24" },
      { label: "Backend", value: "IQM Spark (Odra)" },
      { label: "Avg. Inference", value: "~1.8s" },
      { label: "Accuracy", value: "87.6%" },
      { label: "Training Set", value: "41k spectra" },
    ],
  },
  {
    name: "Classical Random Forest",
    type: "classical",
    description:
      "Classical baseline model using a random forest ensemble. Trained on the same PCA-reduced spectral features as the quantum models. Provides a reference point for evaluating quantum advantage in atmospheric classification tasks.",
    pipeline: [
      "PCA dimensionality reduction (50 → 20 features)",
      "Feature normalization (StandardScaler)",
      "Random Forest (500 trees, max depth 15)",
      "Majority vote ensemble prediction",
      "Calibrated probability output",
    ],
    stats: [
      { label: "Trees", value: "500" },
      { label: "Max Depth", value: "15" },
      { label: "Backend", value: "CPU (scikit-learn)" },
      { label: "Avg. Inference", value: "~100ms" },
      { label: "Accuracy", value: "91.4%" },
      { label: "Training Set", value: "41k spectra" },
    ],
  },
];

const comparisonRows = [
  { label: "Type", values: ["Quantum", "Quantum", "Classical"] },
  { label: "Hardware", values: ["5-qubit IQM", "5-qubit IQM", "CPU"] },
  { label: "Inference Time", values: ["~2.3s", "~1.8s", "~100ms"] },
  { label: "Accuracy", values: ["89.2%", "87.6%", "91.4%"] },
  { label: "Circuit Depth", values: ["12", "24", "N/A"] },
  { label: "Features Used", values: ["5 (PCA)", "5 (PCA)", "20 (PCA)"] },
  { label: "Readout", values: ["Linear SVM", "Ridge Regression", "Random Forest"] },
];

export default function ModelsPage() {
  const ref = useScrollReveal();

  return (
    <div ref={ref}>
      <div className="pt-24 pb-16 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="mb-16">
            <h1 className="reveal text-3xl sm:text-4xl font-bold mb-2">
              Model <span className="gradient-text">Comparison</span>
            </h1>
            <p className="reveal reveal-delay-1 text-muted text-sm max-w-xl">
              Three models trained on the same dataset — two quantum ELMs running on real
              hardware, one classical baseline. Compare architecture, performance, and trade-offs.
            </p>
          </div>

          <div className="space-y-8 mb-20">
            {models.map((m, i) => (
              <div
                key={m.name}
                className={`reveal reveal-delay-${i + 1} p-6 sm:p-8 rounded-xl border border-border bg-surface/50`}
              >
                <div className="flex items-center gap-3 mb-4">
                  {m.type === "quantum" ? (
                    <div className="w-10 h-10 rounded-lg gradient-bg flex items-center justify-center">
                      <Cpu className="w-5 h-5 text-white" />
                    </div>
                  ) : (
                    <div className="w-10 h-10 rounded-lg bg-green/10 border border-green/20 flex items-center justify-center">
                      <TreePine className="w-5 h-5 text-green" />
                    </div>
                  )}
                  <div>
                    <h2 className="text-lg font-bold">{m.name}</h2>
                    <span className="text-xs text-muted font-mono">{m.type}</span>
                  </div>
                </div>

                <p className="text-sm text-muted leading-relaxed mb-6">{m.description}</p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-3">
                      Pipeline
                    </h3>
                    <ol className="space-y-2">
                      {m.pipeline.map((step, j) => (
                        <li key={j} className="flex items-start gap-3 text-sm">
                          <span className="flex-shrink-0 w-5 h-5 rounded-md bg-bg border border-border flex items-center justify-center text-[10px] font-mono text-muted">
                            {j + 1}
                          </span>
                          <span className="text-text/80">{step}</span>
                        </li>
                      ))}
                    </ol>
                  </div>

                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-3">
                      Key Stats
                    </h3>
                    <div className="grid grid-cols-2 gap-3">
                      {m.stats.map((s) => (
                        <div key={s.label}>
                          <div className="text-[10px] text-muted/60 uppercase tracking-wider">
                            {s.label}
                          </div>
                          <div className="text-sm font-mono font-medium">{s.value}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="reveal mb-16">
            <h2 className="text-2xl font-bold mb-6">
              Side-by-side <span className="gradient-text">comparison</span>
            </h2>
            <div className="rounded-xl border border-border overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-surface">
                    <th className="text-left py-3 px-5 text-xs font-semibold text-muted uppercase tracking-wider">
                      Metric
                    </th>
                    {models.map((m) => (
                      <th
                        key={m.name}
                        className="text-left py-3 px-5 text-xs font-semibold text-muted uppercase tracking-wider"
                      >
                        {m.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {comparisonRows.map((row, i) => (
                    <tr
                      key={row.label}
                      className={i % 2 === 0 ? "bg-bg" : "bg-surface/30"}
                    >
                      <td className="py-3 px-5 text-muted font-medium">{row.label}</td>
                      {row.values.map((val, j) => (
                        <td key={j} className="py-3 px-5 font-mono text-xs">
                          {val}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}
