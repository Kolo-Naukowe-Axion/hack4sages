export type Gas = "H2O" | "CO2" | "CO" | "CH4" | "NH3";

export type EvidenceKind = "verified" | "placeholder" | "mixed";

export interface SectionLink {
  id: string;
  label: string;
  index: string;
}

export interface HeroStat {
  label: string;
  value: string;
  tone?: "neutral" | "accent";
}

export interface ModelResult {
  id: string;
  label: string;
  family: string;
  rmse: number;
  trainTime: number;
  complexity: number;
  complexityLabel: string;
  dataEfficiency: number[];
  isVerified: boolean;
}

export interface GasMetric {
  gas: Gas;
  rmse: number;
  mae: number;
  color: string;
  isVerified: boolean;
}

export interface TrendPoint {
  x: number;
  y: number;
}

export interface Series {
  id: string;
  label: string;
  color: string;
  points: TrendPoint[];
  isVerified?: boolean;
}

export interface PipelineStep {
  label: string;
  detail: string;
}
