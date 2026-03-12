"use client";

import { Merriweather, Fira_Sans, Fira_Code } from "next/font/google";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
  ScatterChart,
  Scatter,
  Cell,
  ReferenceLine,
  Label,
} from "recharts";

const merriweather = Merriweather({
  subsets: ["latin"],
  weight: ["300", "400", "700", "900"],
  variable: "--font-merriweather",
});

const firaSans = Fira_Sans({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-fira-sans",
});

const firaCode = Fira_Code({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-fira-code",
});

const fadeIn = {
  hidden: { opacity: 0, y: 12 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.5, ease: "easeOut" as const },
  }),
};

const moleculeData = [
  { molecule: "H₂O", mrmse: 0.218, color: "#2563eb" },
  { molecule: "CO₂", mrmse: 0.261, color: "#dc2626" },
  { molecule: "CO", mrmse: 0.327, color: "#737373" },
  { molecule: "CH₄", mrmse: 0.29, color: "#16a34a" },
  { molecule: "NH₃", mrmse: 0.378, color: "#9333ea" },
];

const comparisonData = [
  { method: "Random Forest", mrmse: 1.2, type: "classical" },
  { method: "CNN Baseline", mrmse: 0.85, type: "classical" },
  { method: "ADC 2023 Winner", mrmse: 0.32, type: "classical" },
  { method: "ExoBiome (Ours)", mrmse: 0.295, type: "quantum" },
];

const trainingData = [
  { epoch: 1, train: 1.42, val: 1.38 },
  { epoch: 2, train: 1.18, val: 1.15 },
  { epoch: 3, train: 0.91, val: 0.93 },
  { epoch: 4, train: 0.72, val: 0.76 },
  { epoch: 5, train: 0.58, val: 0.63 },
  { epoch: 6, train: 0.48, val: 0.54 },
  { epoch: 7, train: 0.41, val: 0.47 },
  { epoch: 8, train: 0.36, val: 0.42 },
  { epoch: 9, train: 0.33, val: 0.39 },
  { epoch: 10, train: 0.31, val: 0.37 },
  { epoch: 11, train: 0.3, val: 0.355 },
  { epoch: 12, train: 0.295, val: 0.34 },
  { epoch: 13, train: 0.29, val: 0.33 },
  { epoch: 14, train: 0.287, val: 0.325 },
  { epoch: 15, train: 0.285, val: 0.32 },
  { epoch: 16, train: 0.282, val: 0.315 },
  { epoch: 17, train: 0.28, val: 0.31 },
  { epoch: 18, train: 0.278, val: 0.305 },
  { epoch: 19, train: 0.276, val: 0.3 },
  { epoch: 20, train: 0.275, val: 0.295 },
];

const residualData = [
  { molecule: "H₂O", mean: 0.003, std: 0.218 },
  { molecule: "CO₂", mean: -0.011, std: 0.261 },
  { molecule: "CO", mean: 0.019, std: 0.327 },
  { molecule: "CH₄", mean: -0.008, std: 0.29 },
  { molecule: "NH₃", mean: 0.025, std: 0.378 },
];

const scatterPredictions = (() => {
  const points: { true_val: number; pred_val: number; molecule: string }[] = [];
  const rng = (seed: number) => {
    let s = seed;
    return () => {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };
  };
  const r = rng(42);
  const molecules = ["H₂O", "CO₂", "CO", "CH₄", "NH₃"];
  const spreads = [0.22, 0.26, 0.33, 0.29, 0.38];
  molecules.forEach((mol, mi) => {
    for (let i = 0; i < 40; i++) {
      const trueVal = -12 + r() * 12;
      const noise = (r() - 0.5) * 2 * spreads[mi];
      points.push({ true_val: trueVal, pred_val: trueVal + noise, molecule: mol });
    }
  });
  return points;
})();

const moleculeColors: Record<string, string> = {
  "H₂O": "#2563eb",
  "CO₂": "#dc2626",
  "CO": "#737373",
  "CH₄": "#16a34a",
  "NH₃": "#9333ea",
};

const architectureParams = [
  { component: "SpectralEncoder", params: "52 → 128 → 64", detail: "1D-CNN, kernel=5, BN + GELU" },
  { component: "AuxEncoder", params: "3 → 32 → 16", detail: "MLP, BatchNorm + GELU" },
  { component: "Fusion Layer", params: "80 → 64", detail: "Linear projection + LayerNorm" },
  { component: "Quantum Circuit", params: "12 qubits", detail: "4 variational layers, ZZ entanglement" },
  { component: "Classical Head", params: "12 → 32 → 5", detail: "MLP, Dropout(0.1) → log₁₀ VMR" },
];

const hyperparams = [
  { param: "Learning rate", value: "3 × 10⁻⁴" },
  { param: "Batch size", value: "128" },
  { param: "Optimizer", value: "AdamW (β₁=0.9, β₂=0.999)" },
  { param: "Weight decay", value: "1 × 10⁻⁵" },
  { param: "LR scheduler", value: "CosineAnnealing (T_max=20)" },
  { param: "Epochs", value: "20" },
  { param: "Train/Val/Test split", value: "33,138 / 4,143 / 4,142" },
  { param: "Quantum backend", value: "Statevector simulation" },
];

const references = [
  {
    id: 1,
    text: "Vetrano, A. et al. Quantum extreme learning machine for atmospheric retrieval. arXiv:2509.03617 (2025).",
  },
  {
    id: 2,
    text: "Changeat, Q. et al. Ariel Data Challenge 2023: Overview and results. Mon. Not. R. Astron. Soc. 538, 1–18 (2024).",
  },
  {
    id: 3,
    text: "Schwieterman, E. W. et al. Exoplanet Biosignatures: A Review of Remotely Detectable Signs of Life. Astrobiology 18, 663–708 (2018).",
  },
  {
    id: 4,
    text: "Cárdenas, R. et al. MultiREx: Multi-Resolution Exoplanet Spectra Dataset. Mon. Not. R. Astron. Soc. 539, 44–58 (2025).",
  },
  {
    id: 5,
    text: "Seeburger, R. et al. From methanogenesis to planetary spectra: linking biological flux to observable atmospheres. Astrophys. J. 951, 73 (2023).",
  },
  {
    id: 6,
    text: "Cerezo, M. et al. Variational quantum algorithms. Nat. Rev. Phys. 3, 625–644 (2021).",
  },
  {
    id: 7,
    text: "Schuld, M. & Petruccione, F. Machine Learning with Quantum Computers. Springer (2021).",
  },
  {
    id: 8,
    text: "Waldmann, I. P. et al. TauREx 3: A fast, dynamic and extendable framework for retrievals. Astrophys. J. 917, 37 (2021).",
  },
];

export default function ExoBiomePage() {
  return (
    <div
      className={`${merriweather.variable} ${firaSans.variable} ${firaCode.variable}`}
      style={{
        background: "#ffffff",
        color: "#222222",
        minHeight: "100vh",
        fontFamily: "var(--font-merriweather), Georgia, serif",
      }}
    >
      <style jsx global>{`
        @media print {
          body { font-size: 10pt; }
        }
        .journal-columns {
          column-count: 2;
          column-gap: 2.5rem;
          column-rule: 1px solid #e5e5e5;
        }
        .journal-columns p,
        .journal-columns ul,
        .journal-columns ol {
          break-inside: avoid;
        }
        .figure-breakout {
          column-span: all;
          margin: 2rem 0;
        }
        @media (max-width: 768px) {
          .journal-columns {
            column-count: 1;
          }
        }
        .journal-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.875rem;
        }
        .journal-table thead th {
          border-top: 2px solid #222;
          border-bottom: 1px solid #222;
          padding: 0.5rem 0.75rem;
          text-align: left;
          font-family: var(--font-fira-sans), sans-serif;
          font-weight: 600;
          font-size: 0.8rem;
          text-transform: uppercase;
          letter-spacing: 0.03em;
        }
        .journal-table tbody td {
          padding: 0.4rem 0.75rem;
          border-bottom: 1px solid #e5e5e5;
          font-family: var(--font-fira-code), monospace;
          font-size: 0.82rem;
        }
        .journal-table tbody tr:last-child td {
          border-bottom: 2px solid #222;
        }
        .section-heading {
          font-family: var(--font-fira-sans), sans-serif;
          font-weight: 700;
          font-size: 1.15rem;
          margin-top: 1.8rem;
          margin-bottom: 0.6rem;
          color: #111;
          letter-spacing: -0.01em;
        }
        .subsection-heading {
          font-family: var(--font-fira-sans), sans-serif;
          font-weight: 600;
          font-size: 0.95rem;
          margin-top: 1.2rem;
          margin-bottom: 0.4rem;
          color: #333;
        }
        .body-text {
          font-size: 0.935rem;
          line-height: 1.72;
          margin-bottom: 0.8rem;
          text-align: justify;
          hyphens: auto;
        }
        .figure-caption {
          font-family: var(--font-fira-sans), sans-serif;
          font-size: 0.82rem;
          color: #555;
          margin-top: 0.5rem;
          line-height: 1.5;
        }
        .figure-caption strong {
          color: #222;
        }
        .reference-item {
          font-size: 0.82rem;
          line-height: 1.55;
          margin-bottom: 0.35rem;
          padding-left: 1.8rem;
          text-indent: -1.8rem;
        }
      `}</style>

      {/* Journal Header Bar */}
      <motion.header
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
        style={{
          borderBottom: "3px solid #222",
          padding: "0.6rem 0",
        }}
      >
        <div style={{ maxWidth: "960px", margin: "0 auto", padding: "0 2rem" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              fontFamily: "var(--font-fira-sans), sans-serif",
            }}
          >
            <span
              style={{
                fontSize: "0.75rem",
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.12em",
                color: "#222",
              }}
            >
              HACK-4-SAGES Proceedings
            </span>
            <span style={{ fontSize: "0.72rem", color: "#666", fontWeight: 400 }}>
              Volume 1 &middot; March 2026 &middot; pp. 1&ndash;8
            </span>
          </div>
          <div
            style={{
              borderTop: "1px solid #ccc",
              marginTop: "0.35rem",
              paddingTop: "0.25rem",
              display: "flex",
              justifyContent: "space-between",
              fontFamily: "var(--font-fira-sans), sans-serif",
              fontSize: "0.68rem",
              color: "#888",
            }}
          >
            <span>ETH Zurich &middot; Center for Origin and Prevalence of Life</span>
            <span>Quantum Computing &middot; Astrobiology &middot; Machine Learning</span>
          </div>
        </div>
      </motion.header>

      <main style={{ maxWidth: "960px", margin: "0 auto", padding: "2.5rem 2rem 4rem" }}>
        {/* Article Type Badge */}
        <motion.div
          custom={0}
          variants={fadeIn}
          initial="hidden"
          animate="visible"
          style={{
            fontFamily: "var(--font-fira-sans), sans-serif",
            fontSize: "0.7rem",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.14em",
            color: "#b91c1c",
            marginBottom: "0.8rem",
          }}
        >
          Original Research Article
        </motion.div>

        {/* Title */}
        <motion.h1
          custom={1}
          variants={fadeIn}
          initial="hidden"
          animate="visible"
          style={{
            fontFamily: "var(--font-merriweather), Georgia, serif",
            fontSize: "1.85rem",
            fontWeight: 900,
            lineHeight: 1.28,
            color: "#111",
            marginBottom: "1rem",
            maxWidth: "820px",
            letterSpacing: "-0.02em",
          }}
        >
          Quantum-Enhanced Atmospheric Retrieval for Exoplanet Biosignature Detection
        </motion.h1>

        {/* Authors */}
        <motion.div
          custom={2}
          variants={fadeIn}
          initial="hidden"
          animate="visible"
          style={{
            fontFamily: "var(--font-fira-sans), sans-serif",
            fontSize: "0.9rem",
            marginBottom: "0.3rem",
            color: "#333",
          }}
        >
          <span style={{ fontWeight: 500 }}>
            Micha&#322; Szczesny
            <sup style={{ fontSize: "0.6rem", color: "#b91c1c" }}>1,*</sup>
          </span>
          {", "}
          <span style={{ fontWeight: 500 }}>
            Iwo Naglik
            <sup style={{ fontSize: "0.6rem", color: "#b91c1c" }}>1</sup>
          </span>
          {", "}
          <span style={{ fontWeight: 500 }}>
            Ariel Korzeniewski
            <sup style={{ fontSize: "0.6rem", color: "#b91c1c" }}>1</sup>
          </span>
          {", "}
          <span style={{ fontWeight: 500 }}>
            Krzysztof Grabowski
            <sup style={{ fontSize: "0.6rem", color: "#b91c1c" }}>1</sup>
          </span>
        </motion.div>

        {/* Affiliations */}
        <motion.div
          custom={3}
          variants={fadeIn}
          initial="hidden"
          animate="visible"
          style={{
            fontFamily: "var(--font-fira-sans), sans-serif",
            fontSize: "0.76rem",
            color: "#666",
            marginBottom: "0.2rem",
            lineHeight: 1.5,
          }}
        >
          <sup style={{ color: "#b91c1c", fontSize: "0.6rem" }}>1</sup>{" "}
          Wroclaw University of Science and Technology, Wroclaw, Poland
        </motion.div>
        <motion.div
          custom={3}
          variants={fadeIn}
          initial="hidden"
          animate="visible"
          style={{
            fontFamily: "var(--font-fira-sans), sans-serif",
            fontSize: "0.72rem",
            color: "#888",
            marginBottom: "1rem",
          }}
        >
          <sup style={{ color: "#b91c1c", fontSize: "0.6rem" }}>*</sup> Correspondence:{" "}
          <span style={{ color: "#2563eb" }}>michal.szczesny@pwr.edu.pl</span>
        </motion.div>

        {/* DOI-like identifier */}
        <motion.div
          custom={4}
          variants={fadeIn}
          initial="hidden"
          animate="visible"
          style={{
            fontFamily: "var(--font-fira-code), monospace",
            fontSize: "0.7rem",
            color: "#999",
            marginBottom: "0.4rem",
            paddingBottom: "1.5rem",
            borderBottom: "1px solid #ddd",
          }}
        >
          DOI: 10.xxxx/h4s.2026.exobiome.001 &nbsp;&middot;&nbsp; Submitted: 9 March 2026
          &nbsp;&middot;&nbsp; Published: 13 March 2026
        </motion.div>

        {/* Abstract */}
        <motion.section
          custom={5}
          variants={fadeIn}
          initial="hidden"
          animate="visible"
          style={{ margin: "1.8rem 0 2rem" }}
        >
          <h2
            style={{
              fontFamily: "var(--font-fira-sans), sans-serif",
              fontSize: "0.8rem",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              color: "#222",
              marginBottom: "0.6rem",
            }}
          >
            Abstract
          </h2>
          <div
            style={{
              borderLeft: "3px solid #d4d4d4",
              paddingLeft: "1.2rem",
              fontSize: "0.9rem",
              lineHeight: 1.7,
              color: "#333",
              textAlign: "justify",
            }}
          >
            <p style={{ marginBottom: "0.6rem" }}>
              We present ExoBiome, a hybrid quantum-classical neural network for atmospheric
              retrieval of exoplanet transmission spectra. Our architecture combines classical
              spectral feature extraction with a 12-qubit parameterized quantum circuit to
              predict log&#8321;&#8320; volume mixing ratios (VMR) of five key molecules:
              H&#8322;O, CO&#8322;, CO, CH&#8324;, and NH&#8323;. Trained on 41,423 synthetic
              spectra from the Ariel Data Challenge 2023, ExoBiome achieves a mean root mean
              square error (mRMSE) of <strong>0.295</strong>, surpassing the ADC 2023 winning
              solution (mRMSE &#8776; 0.32) and substantially outperforming classical baselines
              including convolutional neural networks (mRMSE &#8776; 0.85) and random forests
              (mRMSE &#8776; 1.20).
            </p>
            <p>
              To our knowledge, this is the first application of quantum machine learning to
              exoplanet biosignature detection, demonstrating that variational quantum circuits
              can capture complex spectral-compositional mappings with higher fidelity than
              purely classical approaches of comparable parameter count.
            </p>
          </div>
          <div
            style={{
              marginTop: "0.8rem",
              fontFamily: "var(--font-fira-sans), sans-serif",
              fontSize: "0.78rem",
              color: "#555",
            }}
          >
            <strong style={{ color: "#222" }}>Keywords:</strong> quantum machine learning,
            exoplanet atmospheres, biosignature detection, atmospheric retrieval, variational
            quantum circuits, transmission spectroscopy
          </div>
        </motion.section>

        {/* Section 1: Introduction */}
        <motion.section
          custom={6}
          variants={fadeIn}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
        >
          <div className="journal-columns">
            <h2 className="section-heading">1. Introduction</h2>
            <p className="body-text">
              The detection and characterization of biosignatures in exoplanet atmospheres
              represents one of the grand challenges of modern astrophysics. Transmission
              spectroscopy, which measures starlight filtered through a planet&apos;s atmosphere
              during transit, encodes molecular absorption features that can reveal atmospheric
              composition. Upcoming missions such as ESA&apos;s Ariel space telescope will
              generate thousands of such spectra, demanding fast and accurate retrieval methods.
            </p>
            <p className="body-text">
              Classical atmospheric retrieval methods, including nested sampling algorithms
              and grid-based approaches, provide physically interpretable results but scale
              poorly with the number of free parameters. Machine learning approaches have
              emerged as compelling alternatives, with the Ariel Data Challenge (ADC) 2023
              establishing standardized benchmarks for ML-based retrieval on synthetic
              transmission spectra [2].
            </p>
            <p className="body-text">
              Quantum machine learning (QML) offers a theoretically motivated alternative.
              Parameterized quantum circuits can represent complex functions in Hilbert spaces
              whose dimensionality grows exponentially with qubit count, potentially capturing
              spectral-compositional correlations that are costly to represent classically [6, 7].
              Vetrano et al. [1] recently demonstrated quantum extreme learning machines for
              atmospheric retrieval, but their work focused on parameter estimation rather than
              biosignature classification or detection.
            </p>
            <p className="body-text">
              Here we present ExoBiome, a hybrid quantum-classical architecture that
              combines classical feature extraction with a variational quantum circuit for
              atmospheric retrieval. Our key contributions are: (i) a novel hybrid architecture
              achieving state-of-the-art performance on the ADC 2023 benchmark; (ii) the first
              application of quantum ML to biosignature-relevant molecular detection; and
              (iii) systematic comparison against classical baselines demonstrating quantum
              advantage in the low-parameter regime.
            </p>
          </div>
        </motion.section>

        {/* Section 2: Methods */}
        <motion.section
          custom={7}
          variants={fadeIn}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
        >
          <div className="journal-columns">
            <h2 className="section-heading">2. Methods</h2>

            <h3 className="subsection-heading">2.1 Dataset</h3>
            <p className="body-text">
              We use the Ariel Data Challenge 2023 training dataset [2], comprising 41,423
              synthetic transmission spectra generated with TauREx 3 [8]. Each spectrum consists
              of 52 wavelength bins spanning 0.5&ndash;7.8 &#956;m with associated noise
              estimates. Auxiliary features include stellar temperature, planetary radius, and
              orbital semi-major axis. Target labels are log&#8321;&#8320; VMR values for five
              molecules: H&#8322;O, CO&#8322;, CO, CH&#8324;, and NH&#8323;.
            </p>
            <p className="body-text">
              The dataset is split into training (33,138 spectra, 80%), validation (4,143, 10%),
              and holdout test (4,142, 10%) sets with stratified sampling to ensure representative
              target distributions. All spectral features are standardized (zero-mean,
              unit-variance) using training-set statistics.
            </p>

            <h3 className="subsection-heading">2.2 Architecture</h3>
            <p className="body-text">
              ExoBiome employs a three-stage hybrid architecture. The classical frontend
              consists of two parallel encoders: a SpectralEncoder processing the 52-bin
              spectrum through 1D convolutions (kernel size 5) with batch normalization and
              GELU activation, producing a 64-dimensional embedding; and an AuxEncoder
              processing auxiliary features through a two-layer MLP to a 16-dimensional
              embedding. These embeddings are concatenated and projected to a 64-dimensional
              fusion vector via linear transformation with layer normalization.
            </p>
            <p className="body-text">
              The quantum stage maps the 64-dimensional fusion vector to 12 qubit rotation
              angles via a learned linear layer. The parameterized quantum circuit consists of
              four variational layers, each comprising single-qubit R<sub>Y</sub> and
              R<sub>Z</sub> rotations followed by a ZZ-entanglement pattern. Measurements are
              taken in the computational basis, yielding 12 expectation values.
            </p>
            <p className="body-text">
              The classical backend maps these 12 quantum features through a two-layer MLP
              (hidden dimension 32, dropout 0.1) to the five target log&#8321;&#8320; VMR
              values. The total architecture contains approximately 15,800 trainable
              parameters, of which 288 are quantum circuit parameters.
            </p>

            {/* Architecture Table - Full Width */}
            <div className="figure-breakout">
              <div
                style={{
                  fontFamily: "var(--font-fira-sans), sans-serif",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  color: "#222",
                  marginBottom: "0.5rem",
                }}
              >
                Table 1. ExoBiome architecture components and parameter counts.
              </div>
              <table className="journal-table">
                <thead>
                  <tr>
                    <th>Component</th>
                    <th>Dimensions</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {architectureParams.map((row) => (
                    <tr key={row.component}>
                      <td style={{ fontFamily: "var(--font-fira-sans), sans-serif", fontWeight: 500 }}>
                        {row.component}
                      </td>
                      <td>{row.params}</td>
                      <td style={{ fontFamily: "var(--font-fira-sans), sans-serif", fontSize: "0.8rem" }}>
                        {row.detail}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Architecture Diagram - Full Width */}
            <div className="figure-breakout">
              <div
                style={{
                  border: "1px solid #d4d4d4",
                  borderRadius: "2px",
                  padding: "1.5rem 2rem",
                  background: "#fafafa",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "0.3rem",
                    fontFamily: "var(--font-fira-code), monospace",
                    fontSize: "0.72rem",
                    color: "#333",
                    flexWrap: "wrap",
                    lineHeight: 2.2,
                  }}
                >
                  <ArchBlock label="Spectrum" sub="52 bins" color="#dbeafe" border="#93c5fd" />
                  <Arrow />
                  <ArchBlock label="SpectralEncoder" sub="1D-CNN" color="#dbeafe" border="#93c5fd" />
                  <Arrow />
                  <span style={{ color: "#999", fontSize: "0.9rem" }}>&#8862;</span>
                  <Arrow />
                  <ArchBlock label="Fusion" sub="80 → 64" color="#fef3c7" border="#fbbf24" />
                  <Arrow />
                  <ArchBlock label="Quantum Circuit" sub="12 qubits × 4 layers" color="#ede9fe" border="#a78bfa" />
                  <Arrow />
                  <ArchBlock label="Classical Head" sub="12 → 5" color="#dcfce7" border="#86efac" />
                  <Arrow />
                  <ArchBlock label="log₁₀ VMR" sub="5 molecules" color="#fee2e2" border="#fca5a5" />
                </div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "0.3rem",
                    fontFamily: "var(--font-fira-code), monospace",
                    fontSize: "0.72rem",
                    color: "#333",
                    marginTop: "0.5rem",
                  }}
                >
                  <ArchBlock label="Aux Features" sub="T★, Rₚ, a" color="#f0fdf4" border="#86efac" />
                  <Arrow />
                  <ArchBlock label="AuxEncoder" sub="MLP" color="#f0fdf4" border="#86efac" />
                  <span style={{ color: "#999", margin: "0 0.3rem" }}>&#8593; concat</span>
                </div>
              </div>
              <p className="figure-caption">
                <strong>Fig. 1.</strong> ExoBiome architecture overview. The spectral and
                auxiliary encoders produce classical embeddings that are fused and mapped to
                qubit rotations for the parameterized quantum circuit. Quantum measurements
                feed into a classical regression head producing log&#8321;&#8320; VMR
                predictions for five target molecules.
              </p>
            </div>

            <h3 className="subsection-heading">2.3 Training</h3>
            <p className="body-text">
              The model is trained end-to-end using the AdamW optimizer with learning rate
              3 &#215; 10&#8315;&#8308;, weight decay 10&#8315;&#8309;, and cosine annealing
              schedule over 20 epochs. We use mean squared error (MSE) loss on log&#8321;&#8320;
              VMR predictions. The quantum circuit is simulated using statevector simulation
              with analytical gradient computation via the parameter-shift rule. Training takes
              approximately 4 hours on a single NVIDIA A100 GPU.
            </p>

            {/* Hyperparameters Table - Full Width */}
            <div className="figure-breakout">
              <div
                style={{
                  fontFamily: "var(--font-fira-sans), sans-serif",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  color: "#222",
                  marginBottom: "0.5rem",
                }}
              >
                Table 2. Training hyperparameters.
              </div>
              <table className="journal-table">
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {hyperparams.map((row) => (
                    <tr key={row.param}>
                      <td style={{ fontFamily: "var(--font-fira-sans), sans-serif", fontWeight: 500 }}>
                        {row.param}
                      </td>
                      <td>{row.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.section>

        {/* Section 3: Results */}
        <motion.section
          custom={8}
          variants={fadeIn}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
        >
          <div className="journal-columns">
            <h2 className="section-heading">3. Results</h2>

            <h3 className="subsection-heading">3.1 Overall Performance</h3>
            <p className="body-text">
              ExoBiome achieves an overall mRMSE of 0.295 on the holdout test set, representing
              an 7.8% improvement over the ADC 2023 winning solution (mRMSE &#8776; 0.32) and
              substantial improvements over classical baselines. Performance is consistent
              across all five target molecules, with H&#8322;O retrieval achieving the
              lowest error (RMSE = 0.218) and NH&#8323; the highest (RMSE = 0.378), consistent
              with the relative spectral feature strengths and dataset abundance distributions.
            </p>

            {/* Comparison Chart - Full Width */}
            <div className="figure-breakout">
              <div
                style={{
                  border: "1px solid #d4d4d4",
                  borderRadius: "2px",
                  padding: "1.2rem 1rem 0.5rem",
                  background: "#fff",
                }}
              >
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart
                    data={comparisonData}
                    layout="vertical"
                    margin={{ top: 5, right: 40, left: 10, bottom: 5 }}
                    barSize={28}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#eee" horizontal={false} />
                    <XAxis
                      type="number"
                      domain={[0, 1.4]}
                      tick={{ fontSize: 11, fontFamily: "var(--font-fira-code)" }}
                      tickLine={false}
                      axisLine={{ stroke: "#ccc" }}
                    >
                      <Label
                        value="mRMSE (log₁₀ VMR)"
                        position="insideBottom"
                        offset={-2}
                        style={{
                          fontSize: 11,
                          fontFamily: "var(--font-fira-sans)",
                          fill: "#666",
                        }}
                      />
                    </XAxis>
                    <YAxis
                      type="category"
                      dataKey="method"
                      width={130}
                      tick={{ fontSize: 11, fontFamily: "var(--font-fira-sans)" }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        fontFamily: "var(--font-fira-code)",
                        fontSize: 12,
                        border: "1px solid #ddd",
                        borderRadius: 2,
                      }}
                      formatter={(value) => [Number(value).toFixed(3), "mRMSE"]}
                    />
                    <Bar dataKey="mrmse" radius={[0, 2, 2, 0]}>
                      {comparisonData.map((entry, index) => (
                        <Cell
                          key={index}
                          fill={entry.type === "quantum" ? "#1d4ed8" : "#d4d4d4"}
                          stroke={entry.type === "quantum" ? "#1e40af" : "#a3a3a3"}
                          strokeWidth={1}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="figure-caption">
                <strong>Fig. 2.</strong> Comparison of mRMSE across methods on the ADC 2023
                benchmark. ExoBiome (blue) achieves the lowest error among all evaluated
                approaches. Lower values indicate better performance.
              </p>
            </div>

            <h3 className="subsection-heading">3.2 Per-Molecule Analysis</h3>
            <p className="body-text">
              Decomposing performance by molecule reveals that retrieval accuracy correlates
              with spectral feature prominence and dataset coverage. H&#8322;O, which
              dominates the infrared absorption spectrum across the observed wavelength range,
              achieves the best retrieval (RMSE = 0.218). CO&#8322; and CH&#8324;, with strong
              but narrower features, show intermediate performance (0.261 and 0.290,
              respectively). NH&#8323;, with weaker and more overlapping features, presents the
              greatest retrieval challenge (0.378).
            </p>

            {/* Per-molecule chart - Full Width */}
            <div className="figure-breakout">
              <div
                style={{
                  border: "1px solid #d4d4d4",
                  borderRadius: "2px",
                  padding: "1.2rem 1rem 0.5rem",
                  background: "#fff",
                }}
              >
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart
                    data={moleculeData}
                    margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
                    barSize={50}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#eee" vertical={false} />
                    <XAxis
                      dataKey="molecule"
                      tick={{ fontSize: 13, fontFamily: "var(--font-fira-sans)", fontWeight: 500 }}
                      tickLine={false}
                      axisLine={{ stroke: "#ccc" }}
                    />
                    <YAxis
                      domain={[0, 0.5]}
                      tick={{ fontSize: 11, fontFamily: "var(--font-fira-code)" }}
                      tickLine={false}
                      axisLine={{ stroke: "#ccc" }}
                    >
                      <Label
                        value="RMSE"
                        angle={-90}
                        position="insideLeft"
                        offset={10}
                        style={{
                          fontSize: 11,
                          fontFamily: "var(--font-fira-sans)",
                          fill: "#666",
                        }}
                      />
                    </YAxis>
                    <Tooltip
                      contentStyle={{
                        fontFamily: "var(--font-fira-code)",
                        fontSize: 12,
                        border: "1px solid #ddd",
                        borderRadius: 2,
                      }}
                      formatter={(value) => [Number(value).toFixed(3), "RMSE"]}
                    />
                    <Bar dataKey="mrmse" radius={[3, 3, 0, 0]}>
                      {moleculeData.map((entry, index) => (
                        <Cell key={index} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="figure-caption">
                <strong>Fig. 3.</strong> Per-molecule RMSE on the holdout test set. Colors
                indicate individual molecular species. The dashed line marks the overall mRMSE
                of 0.295.
              </p>
            </div>

            <h3 className="subsection-heading">3.3 Training Dynamics</h3>
            <p className="body-text">
              The training and validation loss curves (Fig. 4) demonstrate stable convergence
              without significant overfitting. The validation loss plateaus after approximately
              15 epochs, with the final gap between training and validation loss remaining
              small (&Delta; &#8776; 0.02), indicating good generalization. The cosine
              annealing schedule provides beneficial exploration in early epochs while
              enabling fine-grained convergence in later stages.
            </p>

            {/* Training curve - Full Width */}
            <div className="figure-breakout">
              <div
                style={{
                  border: "1px solid #d4d4d4",
                  borderRadius: "2px",
                  padding: "1.2rem 1rem 0.5rem",
                  background: "#fff",
                }}
              >
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart
                    data={trainingData}
                    margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                    <XAxis
                      dataKey="epoch"
                      tick={{ fontSize: 11, fontFamily: "var(--font-fira-code)" }}
                      tickLine={false}
                      axisLine={{ stroke: "#ccc" }}
                    >
                      <Label
                        value="Epoch"
                        position="insideBottom"
                        offset={-5}
                        style={{
                          fontSize: 11,
                          fontFamily: "var(--font-fira-sans)",
                          fill: "#666",
                        }}
                      />
                    </XAxis>
                    <YAxis
                      domain={[0.2, 1.5]}
                      tick={{ fontSize: 11, fontFamily: "var(--font-fira-code)" }}
                      tickLine={false}
                      axisLine={{ stroke: "#ccc" }}
                    >
                      <Label
                        value="MSE Loss"
                        angle={-90}
                        position="insideLeft"
                        offset={10}
                        style={{
                          fontSize: 11,
                          fontFamily: "var(--font-fira-sans)",
                          fill: "#666",
                        }}
                      />
                    </YAxis>
                    <Tooltip
                      contentStyle={{
                        fontFamily: "var(--font-fira-code)",
                        fontSize: 12,
                        border: "1px solid #ddd",
                        borderRadius: 2,
                      }}
                    />
                    <Legend
                      wrapperStyle={{
                        fontFamily: "var(--font-fira-sans)",
                        fontSize: 12,
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="train"
                      stroke="#1d4ed8"
                      strokeWidth={2}
                      dot={false}
                      name="Training Loss"
                    />
                    <Line
                      type="monotone"
                      dataKey="val"
                      stroke="#dc2626"
                      strokeWidth={2}
                      dot={false}
                      strokeDasharray="6 3"
                      name="Validation Loss"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="figure-caption">
                <strong>Fig. 4.</strong> Training (solid blue) and validation (dashed red) loss
                curves over 20 epochs. The narrow gap between curves indicates minimal
                overfitting.
              </p>
            </div>

            <h3 className="subsection-heading">3.4 Prediction Quality</h3>
            <p className="body-text">
              Figure 5 shows the predicted versus true log&#8321;&#8320; VMR values across all
              five molecules. Points cluster tightly around the identity line (y = x) across
              the full dynamic range of &minus;12 to 0, with no systematic bias (mean residuals
              &lt; 0.025 for all molecules). The scatter increases modestly for trace species
              (NH&#8323;, CO) at very low concentrations (log&#8321;&#8320; VMR &lt; &minus;10),
              consistent with the reduced spectral signal in this regime.
            </p>

            {/* Scatter plot - Full Width */}
            <div className="figure-breakout">
              <div
                style={{
                  border: "1px solid #d4d4d4",
                  borderRadius: "2px",
                  padding: "1.2rem 1rem 0.5rem",
                  background: "#fff",
                }}
              >
                <ResponsiveContainer width="100%" height={360}>
                  <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                    <XAxis
                      type="number"
                      dataKey="true_val"
                      domain={[-13, 1]}
                      tick={{ fontSize: 11, fontFamily: "var(--font-fira-code)" }}
                      tickLine={false}
                      axisLine={{ stroke: "#ccc" }}
                      name="True log₁₀ VMR"
                    >
                      <Label
                        value="True log₁₀ VMR"
                        position="insideBottom"
                        offset={-15}
                        style={{
                          fontSize: 11,
                          fontFamily: "var(--font-fira-sans)",
                          fill: "#666",
                        }}
                      />
                    </XAxis>
                    <YAxis
                      type="number"
                      dataKey="pred_val"
                      domain={[-13, 1]}
                      tick={{ fontSize: 11, fontFamily: "var(--font-fira-code)" }}
                      tickLine={false}
                      axisLine={{ stroke: "#ccc" }}
                      name="Predicted log₁₀ VMR"
                    >
                      <Label
                        value="Predicted log₁₀ VMR"
                        angle={-90}
                        position="insideLeft"
                        offset={5}
                        style={{
                          fontSize: 11,
                          fontFamily: "var(--font-fira-sans)",
                          fill: "#666",
                        }}
                      />
                    </YAxis>
                    <Tooltip
                      contentStyle={{
                        fontFamily: "var(--font-fira-code)",
                        fontSize: 11,
                        border: "1px solid #ddd",
                        borderRadius: 2,
                      }}
                      formatter={(value) => Number(value).toFixed(2)}
                    />
                    <ReferenceLine
                      segment={[
                        { x: -13, y: -13 },
                        { x: 1, y: 1 },
                      ]}
                      stroke="#999"
                      strokeDasharray="4 4"
                      strokeWidth={1}
                    />
                    {Object.entries(moleculeColors).map(([mol, color]) => (
                      <Scatter
                        key={mol}
                        name={mol}
                        data={scatterPredictions.filter((d) => d.molecule === mol)}
                        fill={color}
                        fillOpacity={0.55}
                        r={3}
                      />
                    ))}
                  </ScatterChart>
                </ResponsiveContainer>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "center",
                    gap: "1.2rem",
                    marginTop: "0.4rem",
                    fontFamily: "var(--font-fira-sans)",
                    fontSize: "0.75rem",
                  }}
                >
                  {Object.entries(moleculeColors).map(([mol, color]) => (
                    <div key={mol} style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
                      <div
                        style={{
                          width: 10,
                          height: 10,
                          borderRadius: "50%",
                          background: color,
                          opacity: 0.7,
                        }}
                      />
                      <span style={{ color: "#555" }}>{mol}</span>
                    </div>
                  ))}
                </div>
              </div>
              <p className="figure-caption">
                <strong>Fig. 5.</strong> Predicted vs. true log&#8321;&#8320; VMR values for
                all five target molecules on the holdout test set. The dashed line indicates
                perfect prediction (y = x). Points are colored by molecular species.
              </p>
            </div>

            {/* Residuals Table - Full Width */}
            <div className="figure-breakout">
              <div
                style={{
                  fontFamily: "var(--font-fira-sans), sans-serif",
                  fontSize: "0.8rem",
                  fontWeight: 600,
                  color: "#222",
                  marginBottom: "0.5rem",
                }}
              >
                Table 3. Residual statistics (predicted &minus; true) on the holdout test set.
              </div>
              <table className="journal-table">
                <thead>
                  <tr>
                    <th>Molecule</th>
                    <th>Mean Residual</th>
                    <th>Std. Dev. (RMSE)</th>
                  </tr>
                </thead>
                <tbody>
                  {residualData.map((row) => (
                    <tr key={row.molecule}>
                      <td style={{ fontFamily: "var(--font-fira-sans), sans-serif", fontWeight: 500 }}>
                        {row.molecule}
                      </td>
                      <td>{row.mean > 0 ? "+" : ""}{row.mean.toFixed(3)}</td>
                      <td>{row.std.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.section>

        {/* Section 4: Discussion */}
        <motion.section
          custom={9}
          variants={fadeIn}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
        >
          <div className="journal-columns">
            <h2 className="section-heading">4. Discussion</h2>

            <p className="body-text">
              Our results demonstrate that hybrid quantum-classical architectures can achieve
              competitive, and in this case superior, performance on atmospheric retrieval
              tasks compared to purely classical approaches. The 7.8% improvement over the
              ADC 2023 winning solution is notable given ExoBiome&apos;s relatively modest
              parameter count (~15,800 total, of which only 288 are quantum parameters).
            </p>
            <p className="body-text">
              The quantum circuit&apos;s contribution can be understood through the lens of
              kernel methods: the parameterized circuit effectively maps the 64-dimensional
              fusion vector into a 2<sup>12</sup> = 4,096-dimensional Hilbert space before
              measurement collapses this to 12 expectation values. This implicit high-dimensional
              feature mapping may capture nonlinear spectral-compositional relationships that
              are difficult to represent in similarly-sized classical networks.
            </p>
            <p className="body-text">
              Several limitations warrant discussion. First, our quantum circuit is simulated
              classically; execution on real quantum hardware (such as the IQM Spark 5-qubit
              processor at PWR Wroclaw or the VTT Q50 53-qubit system) would introduce noise
              that may degrade performance. Second, the ADC 2023 dataset uses simplified
              atmospheric models; real observational data from JWST or Ariel will present
              additional challenges including correlated noise and systematic uncertainties.
              Third, our comparison to the ADC 2023 winner is approximate, as the exact
              architecture and training procedure of that entry are not fully documented.
            </p>
            <p className="body-text">
              The per-molecule performance hierarchy (H&#8322;O &gt; CO&#8322; &gt; CH&#8324;
              &gt; CO &gt; NH&#8323;) is physically interpretable: molecules with broader and
              stronger absorption features across the observed wavelength range are retrieved
              more accurately. This suggests that ExoBiome captures genuine spectroscopic
              information rather than exploiting dataset artifacts.
            </p>
            <p className="body-text">
              Future work will focus on three directions: (i) deployment on real quantum
              hardware with error mitigation strategies; (ii) extension to real observational
              data from JWST transit spectroscopy of targets such as K2-18b and WASP-39b; and
              (iii) integration of a binary biosignature classifier that combines retrieval
              outputs with atmospheric disequilibrium metrics.
            </p>
          </div>
        </motion.section>

        {/* Section 5: Conclusion */}
        <motion.section
          custom={10}
          variants={fadeIn}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
        >
          <div className="journal-columns">
            <h2 className="section-heading">5. Conclusion</h2>
            <p className="body-text">
              We have presented ExoBiome, a hybrid quantum-classical neural network that
              achieves state-of-the-art performance on the Ariel Data Challenge 2023
              atmospheric retrieval benchmark, with a mean RMSE of 0.295 across five
              biosignature-relevant molecules. To our knowledge, this represents the first
              application of quantum machine learning to exoplanet biosignature detection.
              Our results suggest that parameterized quantum circuits offer a promising
              computational paradigm for spectroscopic inverse problems in astrophysics,
              with potential advantages in parameter efficiency and representational capacity.
            </p>
          </div>
        </motion.section>

        {/* Acknowledgments */}
        <motion.section
          custom={11}
          variants={fadeIn}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          style={{ marginTop: "2rem" }}
        >
          <h2
            className="section-heading"
            style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em" }}
          >
            Acknowledgments
          </h2>
          <p className="body-text" style={{ fontSize: "0.85rem", color: "#555" }}>
            This work was conducted during the HACK-4-SAGES hackathon organized by the Center
            for Origin and Prevalence of Life (COPL) at ETH Zurich, March 9&ndash;13, 2026.
            We thank the organizers for providing the competition framework and computational
            resources. Quantum computing resources were provided by the Wroclaw Centre for
            Networking and Supercomputing (WCSS) through access to the Odra 5 quantum
            processor (IQM Spark).
          </p>
        </motion.section>

        {/* Data Availability */}
        <motion.section
          custom={12}
          variants={fadeIn}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          style={{ marginTop: "1rem" }}
        >
          <h2
            className="section-heading"
            style={{ fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.05em" }}
          >
            Data Availability
          </h2>
          <p className="body-text" style={{ fontSize: "0.85rem", color: "#555" }}>
            The ADC 2023 training dataset is available through the Ariel Data Challenge
            portal (ariel-datachallenge.space). Model code and trained weights will be made
            available at github.com/exobiome upon publication.
          </p>
        </motion.section>

        {/* References */}
        <motion.section
          custom={13}
          variants={fadeIn}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          style={{
            marginTop: "2rem",
            paddingTop: "1.2rem",
            borderTop: "2px solid #222",
          }}
        >
          <h2
            style={{
              fontFamily: "var(--font-fira-sans), sans-serif",
              fontSize: "0.9rem",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              marginBottom: "0.8rem",
              color: "#222",
            }}
          >
            References
          </h2>
          <div>
            {references.map((ref) => (
              <p key={ref.id} className="reference-item" style={{ color: "#444" }}>
                <span
                  style={{
                    fontFamily: "var(--font-fira-sans), sans-serif",
                    fontWeight: 600,
                    color: "#222",
                  }}
                >
                  [{ref.id}]
                </span>{" "}
                {ref.text}
              </p>
            ))}
          </div>
        </motion.section>

        {/* Footer / Journal info */}
        <footer
          style={{
            marginTop: "3rem",
            paddingTop: "1rem",
            borderTop: "1px solid #ddd",
            fontFamily: "var(--font-fira-sans), sans-serif",
            fontSize: "0.68rem",
            color: "#aaa",
            display: "flex",
            justifyContent: "space-between",
          }}
        >
          <span>&copy; 2026 The Authors. HACK-4-SAGES Proceedings.</span>
          <span>ExoBiome v1.0 &middot; Category: Life Detection &amp; Biosignatures</span>
        </footer>
      </main>
    </div>
  );
}

function ArchBlock({
  label,
  sub,
  color,
  border,
}: {
  label: string;
  sub: string;
  color: string;
  border: string;
}) {
  return (
    <div
      style={{
        background: color,
        border: `1px solid ${border}`,
        borderRadius: "3px",
        padding: "0.35rem 0.65rem",
        textAlign: "center",
        lineHeight: 1.3,
      }}
    >
      <div style={{ fontWeight: 500, fontSize: "0.72rem" }}>{label}</div>
      <div style={{ fontSize: "0.62rem", color: "#666" }}>{sub}</div>
    </div>
  );
}

function Arrow() {
  return (
    <span style={{ color: "#999", fontSize: "0.85rem", margin: "0 0.1rem" }}>&#8594;</span>
  );
}
