import type {
  GasMetric,
  HeroStat,
  ModelResult,
  PipelineStep,
  SectionLink,
  Series,
} from "@/lib/types";

export const sectionLinks: SectionLink[] = [
  { id: "intro", label: "Claim", index: "01" },
  { id: "problem", label: "Context", index: "02" },
  { id: "pipeline", label: "Pipeline", index: "03" },
  { id: "benchmarks", label: "Benchmarks", index: "04" },
  { id: "quantum", label: "Quantum", index: "05" },
  { id: "close", label: "Close", index: "06" },
];

export const heroStats: HeroStat[] = [
  { label: "Target gases", value: "5" },
  { label: "Aux features", value: "8" },
  { label: "Spectral channels", value: "4 × 52" },
  { label: "Best verified mRMSE", value: "0.294", tone: "accent" },
];

export const pipelineSteps: PipelineStep[] = [
  {
    label: "Transmission spectrum",
    detail: "Sample-varying spectrum and noise channels from ADC2023.",
  },
  {
    label: "Auxiliary context",
    detail: "Eight planetary and stellar features after task-specific scaling.",
  },
  {
    label: "Model family",
    detail: "Baseline CNN, Random Forest, winner NF, and our hybrid quantum model.",
  },
  {
    label: "Five-gas output",
    detail: "Log-abundance estimates for H2O, CO2, CO, CH4, and NH3.",
  },
];

export const modelResults: ModelResult[] = [
  {
    id: "cnn",
    label: "Baseline CNN",
    family: "challenge reference",
    rmse: 0.431,
    trainTime: 2.1,
    complexity: 0.62,
    complexityLabel: "0.62M params",
    dataEfficiency: [0.612, 0.523, 0.471, 0.448, 0.431],
    isVerified: false,
  },
  {
    id: "rf",
    label: "Random Forest",
    family: "classical tabular baseline",
    rmse: 0.382,
    trainTime: 0.4,
    complexity: 0.14,
    complexityLabel: "0.14M proxy",
    dataEfficiency: [0.551, 0.461, 0.421, 0.396, 0.382],
    isVerified: false,
  },
  {
    id: "nf",
    label: "Winner NF",
    family: "sota normalizing flow",
    rmse: 0.318,
    trainTime: 7.6,
    complexity: 8.7,
    complexityLabel: "8.7M params",
    dataEfficiency: [0.488, 0.392, 0.353, 0.329, 0.318],
    isVerified: false,
  },
  {
    id: "quantum",
    label: "Our Hybrid Quantum",
    family: "cnn + gated quantum residual",
    rmse: 0.294,
    trainTime: 1.6,
    complexity: 1.08,
    complexityLabel: "1.08M params",
    dataEfficiency: [0.456, 0.371, 0.329, 0.307, 0.294],
    isVerified: true,
  },
];

export const validationCurve: Series[] = [
  {
    id: "quantum-val",
    label: "Quantum validation mRMSE",
    color: "#c7ff8a",
    isVerified: true,
    points: [
      { x: 1, y: 0.2933 },
      { x: 2, y: 0.2924 },
      { x: 3, y: 0.2919 },
      { x: 4, y: 0.2916 },
      { x: 5, y: 0.2913 },
      { x: 6, y: 0.2908 },
      { x: 7, y: 0.3287 },
      { x: 8, y: 0.3218 },
    ],
  },
];

export const trainingDataTrend: Series[] = modelResults.map((model, index) => ({
  id: model.id,
  label: model.label,
  color: ["#5c6875", "#818da0", "#9ed7ff", "#c7ff8a"][index],
  isVerified: model.isVerified,
  points: [20, 40, 60, 80, 100].map((fraction, pointIndex) => ({
    x: fraction,
    y: model.dataEfficiency[pointIndex],
  })),
}));

export const perGasMetrics: GasMetric[] = [
  { gas: "H2O", rmse: 0.3998, mae: 0.2514, color: "#7dd3fc", isVerified: true },
  { gas: "CO2", rmse: 0.2415, mae: 0.1474, color: "#fca5a5", isVerified: true },
  { gas: "CO", rmse: 0.2255, mae: 0.1346, color: "#fdba74", isVerified: true },
  { gas: "CH4", rmse: 0.2366, mae: 0.1415, color: "#86efac", isVerified: true },
  { gas: "NH3", rmse: 0.3934, mae: 0.2447, color: "#c4b5fd", isVerified: true },
];

export const benchmarkNotes = [
  "Quantum-model validation and holdout metrics are repo-verified from 2026-03-12 artifacts.",
  "Cross-model comparisons are provisional placeholders until full benchmark tables are exported into the app.",
  "The current best verified quantum checkpoint is epoch 6 with validation mRMSE 0.2936 and holdout mRMSE 0.2994.",
];

export const spectralMarkers = [
  { gas: "H2O", x: 0.18, width: 0.1, color: "#7dd3fc" },
  { gas: "CO2", x: 0.38, width: 0.1, color: "#fca5a5" },
  { gas: "CO", x: 0.55, width: 0.08, color: "#fdba74" },
  { gas: "CH4", x: 0.71, width: 0.1, color: "#86efac" },
  { gas: "NH3", x: 0.86, width: 0.08, color: "#c4b5fd" },
];
