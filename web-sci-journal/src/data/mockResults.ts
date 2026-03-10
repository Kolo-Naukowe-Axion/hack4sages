import { PlanetResults, ModelResult, Verdict, DetectedGas } from "@/types";

const gases: Record<string, DetectedGas> = {
  ch4: { formula: "CH₄", name: "Methane", confidence: 0 },
  o3: { formula: "O₃", name: "Ozone", confidence: 0 },
  h2o: { formula: "H₂O", name: "Water", confidence: 0 },
  co2: { formula: "CO₂", name: "Carbon Dioxide", confidence: 0 },
};

function gas(key: string, confidence: number): DetectedGas {
  return { ...gases[key], confidence };
}

function result(
  modelName: string,
  modelType: "quantum" | "classical",
  verdict: Verdict,
  confidence: number,
  detectedGases: DetectedGas[],
  processingTimeMs: number,
  highlights: { start: number; end: number; gas: string }[] = []
): ModelResult {
  return {
    modelName,
    modelType,
    verdict,
    confidence,
    detectedGases,
    processingTimeMs,
    spectrumHighlights: highlights,
  };
}

const defaultHighlights = [
  { start: 1.3, end: 1.5, gas: "H₂O" },
  { start: 3.2, end: 3.5, gas: "CH₄" },
  { start: 4.1, end: 4.5, gas: "CO₂" },
];

const resultsMap: Record<string, [ModelResult, ModelResult, ModelResult]> = {
  "k2-18b": [
    result("QELM Vetrano", "quantum", "detected", 94.2, [gas("ch4", 91.3), gas("co2", 88.7), gas("h2o", 85.1)], 2340, defaultHighlights),
    result("QELM Extended", "quantum", "detected", 91.7, [gas("ch4", 89.2), gas("co2", 86.4), gas("h2o", 82.8)], 1820, defaultHighlights),
    result("Classical RF", "classical", "detected", 96.8, [gas("ch4", 94.1), gas("co2", 91.5), gas("h2o", 88.3)], 120, defaultHighlights),
  ],
  "trappist-1e": [
    result("QELM Vetrano", "quantum", "uncertain", 62.4, [gas("h2o", 71.2), gas("co2", 58.3)], 2510, [{ start: 1.3, end: 1.5, gas: "H₂O" }, { start: 4.1, end: 4.5, gas: "CO₂" }]),
    result("QELM Extended", "quantum", "uncertain", 58.1, [gas("h2o", 65.7), gas("co2", 54.2)], 1930, [{ start: 1.3, end: 1.5, gas: "H₂O" }]),
    result("Classical RF", "classical", "none", 43.2, [gas("co2", 41.8)], 95, [{ start: 4.1, end: 4.5, gas: "CO₂" }]),
  ],
  "lhs-1140b": [
    result("QELM Vetrano", "quantum", "uncertain", 71.3, [gas("h2o", 78.4), gas("co2", 72.1)], 2180, [{ start: 1.3, end: 1.5, gas: "H₂O" }, { start: 4.1, end: 4.5, gas: "CO₂" }]),
    result("QELM Extended", "quantum", "detected", 82.6, [gas("h2o", 84.2), gas("co2", 79.5), gas("ch4", 61.3)], 1750, defaultHighlights),
    result("Classical RF", "classical", "uncertain", 67.9, [gas("h2o", 73.1), gas("co2", 68.4)], 110, [{ start: 1.3, end: 1.5, gas: "H₂O" }, { start: 4.1, end: 4.5, gas: "CO₂" }]),
  ],
  "proxima-cen-b": [
    result("QELM Vetrano", "quantum", "detected", 87.3, [gas("h2o", 82.1), gas("ch4", 76.4), gas("co2", 84.7)], 2420, defaultHighlights),
    result("QELM Extended", "quantum", "detected", 84.9, [gas("h2o", 79.8), gas("ch4", 73.2), gas("co2", 81.5)], 1890, defaultHighlights),
    result("Classical RF", "classical", "detected", 91.2, [gas("h2o", 87.3), gas("ch4", 82.6), gas("co2", 89.1)], 105, defaultHighlights),
  ],
};

function generateDefaultResults(planetId: string): [ModelResult, ModelResult, ModelResult] {
  const seed = planetId.length * 7 + planetId.charCodeAt(0);
  const conf1 = 50 + (seed % 35);
  const conf2 = 45 + ((seed * 3) % 38);
  const conf3 = 55 + ((seed * 5) % 30);

  const v1: Verdict = conf1 > 75 ? "detected" : conf1 > 55 ? "uncertain" : "none";
  const v2: Verdict = conf2 > 75 ? "detected" : conf2 > 55 ? "uncertain" : "none";
  const v3: Verdict = conf3 > 75 ? "detected" : conf3 > 55 ? "uncertain" : "none";

  return [
    result("QELM Vetrano", "quantum", v1, conf1 + (seed % 10) / 10, [gas("h2o", conf1 - 5), gas("co2", conf1 - 12)], 2100 + (seed % 500), [{ start: 1.3, end: 1.5, gas: "H₂O" }, { start: 4.1, end: 4.5, gas: "CO₂" }]),
    result("QELM Extended", "quantum", v2, conf2 + ((seed * 2) % 10) / 10, [gas("h2o", conf2 - 7), gas("co2", conf2 - 10)], 1600 + (seed % 400), [{ start: 1.3, end: 1.5, gas: "H₂O" }]),
    result("Classical RF", "classical", v3, conf3 + ((seed * 4) % 10) / 10, [gas("co2", conf3 - 3), gas("h2o", conf3 - 8)], 80 + (seed % 60), [{ start: 4.1, end: 4.5, gas: "CO₂" }]),
  ];
}

export async function getMockResults(planetId: string): Promise<PlanetResults> {
  const delay = 2000 + Math.random() * 1500;
  await new Promise((resolve) => setTimeout(resolve, delay));

  return {
    planetId,
    results: resultsMap[planetId] ?? generateDefaultResults(planetId),
  };
}
