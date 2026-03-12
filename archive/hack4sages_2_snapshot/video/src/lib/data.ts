// All data from web/presentation-lab/app/lib/project-data.ts on GitHub main

export const gases = [
  { key: "H2O", label: "Water", highlight: "Anchors the interpretation of the other gases." },
  { key: "CO2", label: "Carbon Dioxide", highlight: "Crucial reference gas for disequilibrium analysis." },
  { key: "CO", label: "Carbon Monoxide", highlight: "Useful when methane alone would overstate biology." },
  { key: "CH4", label: "Methane", highlight: "Most compelling when contextualized with CO and CO\u2082." },
  { key: "NH3", label: "Ammonia", highlight: "Makes the five-gas deck feel research-grade." },
];

export const storyline = [
  { title: "Input tensor", detail: "52 wavelength bins \u00d7 4 spectral channels plus 8 auxiliary planetary and stellar features." },
  { title: "Physical preprocessing", detail: "Per-sample spectral normalization, train-split standardization, fixed width and wavelength templates." },
  { title: "Hybrid predictor", detail: "Residual 1D encoder, attention pooling, classical regression head, and an 8-qubit quantum correction branch." },
  { title: "Scientific output", detail: "Joint abundance estimates for H\u2082O, CO\u2082, CO, CH\u2084, and NH\u2083, interpreted as biosignature evidence only in combination." },
];

export const projectFacts = [
  { label: "Dataset", value: "Ariel Data Challenge 2023" },
  { label: "Train / Val / Holdout", value: "33,138 / 4,142 / 4,143" },
  { label: "Targets", value: "5 gas abundances" },
  { label: "Auxiliary features", value: "8" },
];

export const performance = {
  bestEpoch: 6,
  bestTrainingVal: 0.29081112146377563,
  validation: 0.29361358284950256,
  holdout: 0.2993761897087097,
  holdoutMae: 0.18392649292945862,
  rows: 4143,
  perGasRmse: [
    { gas: "CH\u2084", key: "CH4", value: 0.236580029129982 },
    { gas: "CO", key: "CO", value: 0.2255290001630783 },
    { gas: "CO\u2082", key: "CO2", value: 0.24153712391853333 },
    { gas: "H\u2082O", key: "H2O", value: 0.39978623390197754 },
    { gas: "NH\u2083", key: "NH3", value: 0.3934485614299774 },
  ],
};

export const modelRoster = [
  { name: "Organizer Baseline CNN", className: "Classical CNN", position: "challenge reference" },
  { name: "Random Forest", className: "Classical ensemble", position: "robust baseline" },
  { name: "Winner-Style NSF", className: "Challenge-winning family", position: "state-of-the-art reference" },
  { name: "Hybrid Quantum Residual", className: "Custom quantum ML", position: "team highlight" },
];

// --- Spectrum data (52 bins, 0.5–5.0 μm) ---
function gaussian(x: number, mean: number, spread: number, amplitude: number) {
  const distance = x - mean;
  return amplitude * Math.exp(-(distance * distance) / (2 * spread * spread));
}

const wavelengthCount = 52;

export const spectrumSeries = Array.from({ length: wavelengthCount }, (_, index) => {
  const wavelength = 0.5 + index * ((5.0 - 0.5) / (wavelengthCount - 1));
  const baseline = 1.03 - index * 0.0032;
  const absorption =
    gaussian(wavelength, 1.4, 0.13, 0.12) +
    gaussian(wavelength, 2.7, 0.18, 0.09) +
    gaussian(wavelength, 4.3, 0.14, 0.17);
  const ripple = Math.sin(wavelength * 5.8) * 0.018 + Math.cos(wavelength * 2.6) * 0.008;
  return {
    wavelength,
    observed: Number((baseline - absorption + ripple).toFixed(4)),
    retrieved: Number((baseline - absorption * 0.93 + ripple * 0.45 + 0.007).toFixed(4)),
  };
});
