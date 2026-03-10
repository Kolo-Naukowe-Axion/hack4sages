"use client";

import { StarField } from "@/components/StarField";
import { ModelCard } from "@/components/ModelCard";
import { ComparisonTable } from "@/components/ComparisonTable";
import { useScrollReveal } from "@/hooks/useScrollReveal";
import { Cpu } from "lucide-react";

const models = [
  {
    name: "QELM Vetrano",
    type: "quantum" as const,
    description:
      "Faithful reproduction of Vetrano et al. 2025 (arXiv:2509.03617). Uses angle encoding to map spectral features into qubit rotations, then measures a quantum random feature map. The reservoir's random projections are read out and fed into a ridge regression classifier.",
    pipeline: [
      { label: "PCA Reduction", detail: "50 spectral points to 5 principal components" },
      { label: "Angle Encoding", detail: "Each PC mapped to RY rotation on one qubit" },
      { label: "Random Reservoir", detail: "Fixed random unitary on IQM Spark (5 qubits)" },
      { label: "Measurement", detail: "Expectation values of Pauli-Z on all qubits" },
      { label: "Ridge Classifier", detail: "Linear readout trained on 500 labeled spectra" },
    ],
    stats: [
      { label: "Qubits", value: "5" },
      { label: "Hardware", value: "IQM Spark" },
      { label: "Encoding", value: "Angle (RY)" },
      { label: "Training Set", value: "500 spectra" },
    ],
  },
  {
    name: "QELM Extended",
    type: "quantum" as const,
    description:
      "Extended variant using amplitude encoding for richer quantum feature maps. Encodes normalized spectral vectors into quantum state amplitudes, enabling the reservoir to explore a larger Hilbert space with the same 5 qubits.",
    pipeline: [
      { label: "PCA Reduction", detail: "50 spectral points to 5 principal components" },
      { label: "Amplitude Encoding", detail: "Normalized PC vector as quantum state amplitudes" },
      { label: "Entangling Layer", detail: "CNOT cascade + random rotations on IQM Spark" },
      { label: "Measurement", detail: "Full tomography of 2-qubit reduced density matrices" },
      { label: "Ridge Classifier", detail: "Linear readout on expanded feature vector" },
    ],
    stats: [
      { label: "Qubits", value: "5" },
      { label: "Hardware", value: "IQM Spark" },
      { label: "Encoding", value: "Amplitude" },
      { label: "Training Set", value: "500 spectra" },
    ],
  },
  {
    name: "Classical RF",
    type: "classical" as const,
    description:
      "Random Forest baseline for comparison. Uses scikit-learn's ensemble of 200 decision trees trained on a larger dataset. Provides a speed and accuracy reference against quantum approaches.",
    pipeline: [
      { label: "PCA Reduction", detail: "50 spectral points to 20 principal components" },
      { label: "Feature Engineering", detail: "Band ratios, slope features, peak detection" },
      { label: "Random Forest", detail: "200 trees, max depth 15, Gini impurity" },
      { label: "Ensemble Vote", detail: "Majority voting across all trees" },
      { label: "Probability Output", detail: "Class probability from vote fractions" },
    ],
    stats: [
      { label: "Estimators", value: "200" },
      { label: "Hardware", value: "CPU" },
      { label: "Max Depth", value: "15" },
      { label: "Training Set", value: "10,000 spectra" },
    ],
  },
];

export default function ModelsPage() {
  const scrollRef = useScrollReveal();

  return (
    <div ref={scrollRef} className="relative min-h-screen pt-16">
      <StarField />

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-12">
        <div className="flex items-center gap-3 mb-2">
          <Cpu className="w-5 h-5 text-cyan" />
          <h1 className="text-2xl font-bold text-white">Models</h1>
        </div>
        <p className="text-sm text-white/40 mb-10 max-w-2xl">
          Three models for atmospheric biosignature classification: two quantum
          (QELM) running on real IQM Spark hardware, and one classical Random
          Forest baseline.
        </p>

        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {models.map((m) => (
            <div key={m.name} className="reveal">
              <ModelCard {...m} />
            </div>
          ))}
        </div>

        <div className="reveal">
          <ComparisonTable />
        </div>
      </div>
    </div>
  );
}
