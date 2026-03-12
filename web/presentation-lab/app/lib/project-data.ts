export type GasProfile = {
  key: "H2O" | "CO2" | "CO" | "CH4" | "NH3";
  label: string;
  note: string;
  role: string;
  highlight: string;
};

export type StoryStep = {
  title: string;
  detail: string;
};

export type RouteVariant = {
  slug: string;
  title: string;
  framing: string;
  description: string;
};

const wavelengthCount = 52;

function gaussian(x: number, mean: number, spread: number, amplitude: number) {
  const distance = x - mean;
  return amplitude * Math.exp(-(distance * distance) / (2 * spread * spread));
}

export const gases: GasProfile[] = [
  {
    key: "H2O",
    label: "Water",
    note: "Stable atmospheric context and habitability signal.",
    role: "Benchmark volatile in transmission spectra.",
    highlight: "Anchors the interpretation of the other gases."
  },
  {
    key: "CO2",
    label: "Carbon Dioxide",
    note: "Greenhouse and carbon-cycle context.",
    role: "Controls atmospheric chemistry assumptions.",
    highlight: "Crucial reference gas for disequilibrium analysis."
  },
  {
    key: "CO",
    label: "Carbon Monoxide",
    note: "Counterweight to methane-rich interpretations.",
    role: "Helps reject false-positive chemistry stories.",
    highlight: "Useful when methane alone would overstate biology."
  },
  {
    key: "CH4",
    label: "Methane",
    note: "Classic reduced-gas biosignature candidate.",
    role: "Highly visible in many retrieval discussions.",
    highlight: "Most compelling when contextualized with CO and CO2."
  },
  {
    key: "NH3",
    label: "Ammonia",
    note: "High-value trace gas with difficult retrieval dynamics.",
    role: "Expands beyond the most common exoplanet demos.",
    highlight: "Makes the five-gas deck feel research-grade."
  }
];

export const storyline: StoryStep[] = [
  {
    title: "Input tensor",
    detail: "52 wavelength bins x 4 spectral channels plus 8 auxiliary planetary and stellar features."
  },
  {
    title: "Physical preprocessing",
    detail: "Per-sample spectral normalization, train-split standardization, fixed width and wavelength templates."
  },
  {
    title: "Hybrid predictor",
    detail: "Residual 1D encoder, attention pooling, classical regression head, and an 8-qubit quantum correction branch."
  },
  {
    title: "Scientific output",
    detail: "Joint abundance estimates for H2O, CO2, CO, CH4, and NH3, interpreted as biosignature evidence only in combination."
  }
];

export const projectFacts = [
  { label: "Dataset", value: "Ariel Data Challenge 2023" },
  { label: "Train / Val / Holdout", value: "33,138 / 4,142 / 4,143" },
  { label: "Targets", value: "5 gas abundances" },
  { label: "Auxiliary features", value: "8" }
];

export const performance = {
  bestEpoch: 6,
  bestTrainingVal: 0.29081112146377563,
  validation: 0.29361358284950256,
  holdout: 0.2993761897087097,
  holdoutMae: 0.18392649292945862,
  rows: 4143,
  perGasRmse: [
    { gas: "CH4", value: 0.236580029129982 },
    { gas: "CO", value: 0.2255290001630783 },
    { gas: "CO2", value: 0.24153712391853333 },
    { gas: "H2O", value: 0.39978623390197754 },
    { gas: "NH3", value: 0.3934485614299774 }
  ]
};

export const modelRoster = [
  {
    name: "Organizer Baseline CNN",
    className: "Classical CNN",
    position: "challenge reference",
    summary: "Canonical starting point from the Ariel challenge ecosystem."
  },
  {
    name: "Random Forest",
    className: "Classical ensemble",
    position: "robust baseline",
    summary: "Strong non-neural baseline for tabular plus spectral aggregation."
  },
  {
    name: "Winner-Style NSF",
    className: "Challenge-winning family",
    position: "state-of-the-art reference",
    summary: "Flow-based retrieval route derived from the challenge winner."
  },
  {
    name: "Hybrid Quantum Residual",
    className: "Custom quantum ML",
    position: "team highlight",
    summary: "Best verified checkpoint in this repo reaches 0.2994 holdout mRMSE."
  }
];

export const routeVariants: RouteVariant[] = [
  {
    slug: "1",
    title: "Swiss Control Sheet",
    framing: "annual-report discipline",
    description: "A bright, controlled presentation sheet driven by alignment, rules, and hard editorial hierarchy."
  },
  {
    slug: "2",
    title: "CRT Spectrograph",
    framing: "instrument terminal",
    description: "A monochrome phosphor observatory display that reads like a live terminal transcript."
  },
  {
    slug: "3",
    title: "Solar Atlas",
    framing: "scientific map plate",
    description: "A radial atlas composition that organizes the whole story around one annotated celestial figure."
  },
  {
    slug: "4",
    title: "Art Deco Observatory",
    framing: "ceremonial observatory program",
    description: "A symmetrical black-and-gold presentation with marquee framing and formal scientific pageantry."
  },
  {
    slug: "5",
    title: "Broadsheet Review",
    framing: "scientific weekly",
    description: "A newspaper-driven layout built from columns, pull quotes, lead stories, and one decisive comparison table."
  },
  {
    slug: "6",
    title: "Monolith Keynote",
    framing: "live talk deck",
    description: "A sequence of giant full-height statements for a two-minute pitch with almost no small-component language."
  },
  {
    slug: "7",
    title: "Blueprint Roll",
    framing: "engineering drawing",
    description: "A cyan-on-blue technical sheet that explains the system as one continuous mechanism."
  },
  {
    slug: "8",
    title: "Cabinet of Specimens",
    framing: "archive drawers",
    description: "A warm natural-history archive that presents the gases and models as labeled scientific specimens."
  }
];

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
    retrieved: Number((baseline - absorption * 0.93 + ripple * 0.45 + 0.007).toFixed(4))
  };
});
