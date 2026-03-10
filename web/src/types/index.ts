export interface Planet {
  id: string;
  name: string;
  starSystem: string;
  discoveryYear: number;
  massEarth: number | null;
  radiusEarth: number;
  eqTempK: number | null;
  orbitalPeriodDays: number;
  distanceLy: number;
  inHabitableZone: boolean;
  hasJWSTData: boolean;
  spectrumType: "jwst" | "synthetic";
  spectrumData: { wavelength: number; flux: number }[];
}

export type Verdict = "detected" | "none" | "uncertain";

export interface DetectedGas {
  formula: string;
  name: string;
  confidence: number;
}

export interface ModelResult {
  modelName: string;
  modelType: "quantum" | "classical";
  verdict: Verdict;
  confidence: number;
  detectedGases: DetectedGas[];
  processingTimeMs: number;
  spectrumHighlights: { start: number; end: number; gas: string }[];
}

export interface PlanetResults {
  planetId: string;
  results: [ModelResult, ModelResult, ModelResult];
}
