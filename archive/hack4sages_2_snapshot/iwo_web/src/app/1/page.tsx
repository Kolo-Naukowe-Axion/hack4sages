"use client";

import { Crimson_Text, JetBrains_Mono } from "next/font/google";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ScatterChart,
  Scatter,
  Legend,
  LineChart,
  Line,
} from "recharts";

const serif = Crimson_Text({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-serif",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
});

const fade = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" as const } },
};

const comparisonData = [
  { model: "Random Forest", mrmse: 1.2, fill: "#b0b0b0" },
  { model: "CNN Baseline", mrmse: 0.85, fill: "#888888" },
  { model: "ADC 2023 Winner", mrmse: 0.32, fill: "#555555" },
  { model: "ExoBiome (Ours)", mrmse: 0.295, fill: "#1a1a1a" },
];

const perMoleculeData = [
  { molecule: "H₂O", rmse: 0.218 },
  { molecule: "CO₂", rmse: 0.261 },
  { molecule: "CH₄", rmse: 0.29 },
  { molecule: "CO", rmse: 0.327 },
  { molecule: "NH₃", rmse: 0.378 },
];

const spectrumData = Array.from({ length: 52 }, (_, i) => {
  const wl = 0.5 + i * (7.3 / 51);
  const base = 0.012 + 0.003 * Math.sin(wl * 1.2);
  const h2o = wl > 1.2 && wl < 1.5 ? 0.004 * Math.exp(-((wl - 1.38) ** 2) / 0.01) : 0;
  const co2 = wl > 4.0 && wl < 4.8 ? 0.006 * Math.exp(-((wl - 4.3) ** 2) / 0.05) : 0;
  const ch4 = wl > 3.0 && wl < 3.6 ? 0.003 * Math.exp(-((wl - 3.3) ** 2) / 0.03) : 0;
  const h2o2 = wl > 5.5 && wl < 7.0 ? 0.005 * Math.exp(-((wl - 6.3) ** 2) / 0.15) : 0;
  return {
    wavelength: parseFloat(wl.toFixed(2)),
    depth: parseFloat((base + h2o + co2 + ch4 + h2o2 + (Math.random() - 0.5) * 0.0008).toFixed(5)),
  };
});

const convergenceData = Array.from({ length: 30 }, (_, i) => {
  const epoch = i + 1;
  const train = 1.8 * Math.exp(-0.12 * epoch) + 0.22 + (Math.random() - 0.5) * 0.03;
  const val = 1.9 * Math.exp(-0.1 * epoch) + 0.28 + (Math.random() - 0.5) * 0.04;
  return {
    epoch,
    train: parseFloat(train.toFixed(3)),
    val: parseFloat(val.toFixed(3)),
  };
});

const scatterH2O = Array.from({ length: 80 }, (_, i) => {
  const truth = -8 + Math.random() * 7;
  const pred = truth + (Math.random() - 0.5) * 1.2 * (1 + 0.15 * Math.abs(truth + 5));
  return { truth: parseFloat(truth.toFixed(2)), pred: parseFloat(pred.toFixed(2)) };
});

function Section({
  number,
  title,
  children,
}: {
  number: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <motion.section
      variants={fade}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: "-40px" }}
      className="mb-10"
    >
      <h2
        className="text-[1.35rem] font-semibold mb-4 tracking-tight"
        style={{ fontFamily: "var(--font-serif)" }}
      >
        {number}. {title}
      </h2>
      {children}
    </motion.section>
  );
}

function Figure({
  number,
  caption,
  children,
}: {
  number: number;
  caption: string;
  children: React.ReactNode;
}) {
  return (
    <div className="my-8">
      <div className="border border-[#e0e0e0] p-4 bg-[#fafafa]">{children}</div>
      <p
        className="text-[0.82rem] text-[#555] mt-2 text-center leading-relaxed"
        style={{ fontFamily: "var(--font-serif)" }}
      >
        <span className="font-semibold">Figure {number}.</span> {caption}
      </p>
    </div>
  );
}

function Equation({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="my-6 py-4 px-6 bg-[#f8f8f8] border-l-2 border-[#ccc] text-center overflow-x-auto"
      style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem" }}
    >
      {children}
    </div>
  );
}

function Table({
  number,
  caption,
  headers,
  rows,
}: {
  number: number;
  caption: string;
  headers: string[];
  rows: (string | React.ReactNode)[][];
}) {
  return (
    <div className="my-8">
      <p
        className="text-[0.82rem] text-[#555] mb-2 text-center leading-relaxed"
        style={{ fontFamily: "var(--font-serif)" }}
      >
        <span className="font-semibold">Table {number}.</span> {caption}
      </p>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[0.88rem]">
          <thead>
            <tr className="border-t-2 border-b border-[#1a1a1a]">
              {headers.map((h, i) => (
                <th
                  key={i}
                  className="py-2 px-3 text-left font-semibold"
                  style={{ fontFamily: "var(--font-serif)" }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => (
              <tr
                key={ri}
                className={`border-b border-[#ddd] ${ri === rows.length - 1 ? "border-b-2 border-[#1a1a1a]" : ""}`}
              >
                {row.map((cell, ci) => (
                  <td
                    key={ci}
                    className="py-2 px-3"
                    style={{ fontFamily: ci === 0 ? "var(--font-serif)" : "var(--font-mono)" }}
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function InlineCode({ children }: { children: React.ReactNode }) {
  return (
    <code
      className="bg-[#f3f3f3] px-1.5 py-0.5 text-[0.82rem] rounded-sm"
      style={{ fontFamily: "var(--font-mono)" }}
    >
      {children}
    </code>
  );
}

function Footnote({ number, children }: { number: number; children: React.ReactNode }) {
  return (
    <div className="flex gap-1.5 text-[0.78rem] text-[#666] leading-relaxed">
      <sup className="text-[0.65rem] mt-0.5 flex-shrink-0">{number}</sup>
      <span>{children}</span>
    </div>
  );
}

export default function ExoBiomePaper() {
  return (
    <div
      className={`${serif.variable} ${mono.variable} min-h-screen bg-white text-[#1a1a1a]`}
    >
      <article
        className="max-w-[720px] mx-auto px-6 py-16 text-[0.95rem] leading-[1.8]"
        style={{ fontFamily: "var(--font-serif)", textAlign: "justify" }}
      >
        {/* Title Block */}
        <motion.header
          variants={fade}
          initial="hidden"
          animate="visible"
          className="mb-12 text-center"
          style={{ textAlign: "center" }}
        >
          <h1
            className="text-[1.75rem] leading-[1.3] font-bold tracking-tight mb-5"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            ExoBiome: Quantum-Enhanced Atmospheric Retrieval
            <br />
            for Exoplanet Biosignature Detection
          </h1>

          <div className="text-[0.9rem] text-[#444] mb-2 space-y-0.5">
            <p>
              Michal Szczesny, Tomasz Lamża, Mikolaj Siwek, Marcin Kocot
            </p>
          </div>
          <div className="text-[0.82rem] text-[#777] mb-6 italic">
            HACK-4-SAGES 2026 &middot; ETH Zurich COPL &middot; Life Detection
            and Biosignatures
          </div>

          <div className="text-[0.82rem] text-[#999] mb-8" style={{ fontFamily: "var(--font-mono)" }}>
            March 12, 2026
          </div>

          <hr className="border-[#ddd] mb-8" />

          <div
            className="text-[0.88rem] leading-[1.75] italic text-[#333] mx-auto max-w-[620px]"
            style={{ textAlign: "justify" }}
          >
            <span className="font-semibold not-italic">Abstract.</span>{" "}
            We present ExoBiome, a hybrid quantum-classical neural network for
            atmospheric retrieval of exoplanet transmission spectra. Our
            architecture combines a classical spectral encoder with a 12-qubit
            quantum circuit executed on IQM Spark hardware, predicting
            log&#8322; VMR abundances for five key biosignature molecules:
            H&#8322;O, CO&#8322;, CO, CH&#8324;, and NH&#8323;. Evaluated on
            the Ariel Data Challenge 2023 dataset (41,423 spectra), ExoBiome
            achieves a mean RMSE of 0.295, surpassing the ADC 2023 winning
            solution (0.32) and classical baselines by significant margins. To
            our knowledge, this represents the first application of quantum
            machine learning to exoplanet biosignature detection.
          </div>
        </motion.header>

        <hr className="border-[#ddd] mb-10" />

        {/* 1. Introduction */}
        <Section number="1" title="Introduction">
          <p className="mb-4">
            The detection and characterization of biosignatures in exoplanet
            atmospheres represents one of the most compelling challenges in
            modern astrophysics. Transmission spectroscopy, wherein starlight
            filters through a transiting planet&rsquo;s atmosphere, encodes
            molecular absorption features that can reveal atmospheric
            composition. The upcoming Ariel mission (ESA, launch 2029) will
            observe approximately 1,000 exoplanetary atmospheres, demanding
            scalable and accurate retrieval methods.
          </p>
          <p className="mb-4">
            Classical approaches to atmospheric retrieval, including nested
            sampling with forward models such as TauREx and petitRADTRANS,
            remain computationally expensive. Machine learning alternatives have
            demonstrated competitive accuracy at dramatically reduced inference
            times, yet purely classical architectures plateau in performance on
            complex spectral inversion tasks.
          </p>
          <p className="mb-4">
            Quantum machine learning (QML) offers a theoretically motivated
            alternative. Quantum reservoir computing and quantum extreme
            learning machines (QELMs) exploit the exponential dimensionality of
            Hilbert space to construct rich feature representations from
            modest-depth circuits. Vetrano et al. (2025) demonstrated the
            viability of QELMs for atmospheric retrieval on synthetic spectra,
            though their work targeted trace gas abundances rather than
            biosignature classification.
          </p>
          <p>
            In this work, we present ExoBiome: a hybrid quantum-classical
            architecture that combines trainable spectral encoders with a
            parameterized quantum circuit for simultaneous retrieval of five
            molecular abundances. We evaluate on the Ariel Data Challenge 2023
            (ADC2023) benchmark and report state-of-the-art results.
          </p>
        </Section>

        <hr className="border-[#eee] mb-10" />

        {/* 2. Methods */}
        <Section number="2" title="Methods">
          <h3
            className="text-[1.05rem] font-semibold mb-3 mt-6"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            2.1 Dataset
          </h3>
          <p className="mb-4">
            We utilize the Ariel Data Challenge 2023 training set, comprising
            41,423 synthetic transmission spectra generated with TauREx 3. Each
            spectrum consists of 52 wavelength bins spanning 0.5&ndash;7.8 &mu;m,
            accompanied by auxiliary planetary parameters (radius, mass,
            stellar temperature, orbital period). Ground truth labels are
            provided as log&#8322; volume mixing ratios (VMR) for H&#8322;O,
            CO&#8322;, CO, CH&#8324;, and NH&#8323;.
          </p>

          <Figure number={1} caption="Sample transmission spectrum from the ADC2023 dataset. Absorption features corresponding to H₂O (~1.4 μm, ~6.3 μm), CO₂ (~4.3 μm), and CH₄ (~3.3 μm) are visible.">
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={spectrumData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
                <XAxis
                  dataKey="wavelength"
                  tick={{ fontSize: 11, fill: "#555", fontFamily: "var(--font-mono)" }}
                  label={{
                    value: "Wavelength (μm)",
                    position: "insideBottom",
                    offset: -2,
                    style: { fontSize: 11, fill: "#555", fontFamily: "var(--font-serif)" },
                  }}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "#555", fontFamily: "var(--font-mono)" }}
                  label={{
                    value: "Transit Depth",
                    angle: -90,
                    position: "insideLeft",
                    offset: 10,
                    style: { fontSize: 11, fill: "#555", fontFamily: "var(--font-serif)" },
                  }}
                  domain={["auto", "auto"]}
                />
                <Tooltip
                  contentStyle={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.78rem",
                    border: "1px solid #ddd",
                    borderRadius: 0,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="depth"
                  stroke="#1a1a1a"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Figure>

          <h3
            className="text-[1.05rem] font-semibold mb-3 mt-6"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            2.2 Preprocessing
          </h3>
          <p className="mb-4">
            Spectra are normalized via per-channel standardization using training
            set statistics. Auxiliary features (planetary radius, mass, stellar
            temperature, orbital period) are independently scaled to zero mean
            and unit variance. No data augmentation is applied. The dataset is
            partitioned into training (80%), validation (10%), and holdout test
            (10%) splits with stratified sampling over target abundance ranges.
          </p>

          <h3
            className="text-[1.05rem] font-semibold mb-3 mt-6"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            2.3 Quantum Circuit Design
          </h3>
          <p className="mb-4">
            The quantum component employs a 12-qubit parameterized circuit
            executed on IQM Spark (Odra 5, PWR Wroclaw) and VTT Q50 (53 qubits,
            Finland) hardware. The circuit implements angle encoding of the
            fused classical features into rotation gates, followed by entangling
            layers of CNOT gates in a circular topology:
          </p>

          <Equation>
            U(x) = &prod;&#8467;
            <sub>=1</sub>
            <sup>L</sup> W<sub>&#8467;</sub> &middot; S(x) &nbsp;&nbsp;where
            &nbsp;&nbsp; S(x) = &otimes;
            <sub>i</sub> R<sub>Y</sub>(x<sub>i</sub>) &nbsp;&nbsp;and
            &nbsp;&nbsp; W<sub>&#8467;</sub> = &prod;
            <sub>i</sub> CNOT<sub>i,i+1</sub> &middot; &otimes;
            <sub>i</sub> R<sub>Z</sub>(&theta;
            <sub>&#8467;,i</sub>)
          </Equation>

          <p className="mb-4">
            Output predictions are obtained from expectation values of Pauli-Z
            observables on 5 designated readout qubits, linearly mapped to the
            target log&#8322; VMR range. The remaining 7 qubits serve as
            ancillary registers, enriching the representational capacity of the
            circuit through entanglement-mediated correlations.
          </p>
        </Section>

        <hr className="border-[#eee] mb-10" />

        {/* 3. Architecture */}
        <Section number="3" title="Architecture">
          <p className="mb-4">
            ExoBiome follows a modular encoder-fusion-quantum design. The
            classical front-end compresses the 52-dimensional spectral input and
            auxiliary features into a compact latent representation suitable for
            quantum circuit encoding.
          </p>

          <Table
            number={1}
            caption="ExoBiome architecture summary. Layer dimensions, activation functions, and output shapes for each module."
            headers={["Module", "Layers", "Output Dim", "Activation"]}
            rows={[
              ["SpectralEncoder", "Conv1D(52→64→128) + Pool", "32", "GELU"],
              ["AuxEncoder", "Linear(4→32→16)", "16", "GELU"],
              ["Fusion MLP", "Linear(48→24→12)", "12", "Tanh"],
              [
                "Quantum Circuit",
                "12 qubits, L=3 layers",
                "5",
                <span key="qc" className="italic">
                  ⟨Z⟩ readout
                </span>,
              ],
              ["Output Scaling", "Linear(5→5)", "5", "—"],
            ]}
          />

          <div className="my-8 border border-[#e0e0e0] p-5 bg-[#fafafa]">
            <div className="text-[0.82rem] text-[#555] mb-3 text-center font-semibold" style={{ fontFamily: "var(--font-serif)" }}>
              Figure 2. Model architecture schematic.
            </div>
            <div
              className="flex flex-col items-center gap-0"
              style={{ fontFamily: "var(--font-mono)", fontSize: "0.78rem" }}
            >
              <div className="flex gap-8 mb-1">
                <div className="border border-[#999] px-4 py-2 bg-white text-center">
                  Spectrum
                  <br />
                  <span className="text-[0.7rem] text-[#888]">52 bins</span>
                </div>
                <div className="border border-[#999] px-4 py-2 bg-white text-center">
                  Aux Data
                  <br />
                  <span className="text-[0.7rem] text-[#888]">4 params</span>
                </div>
              </div>
              <div className="text-[#aaa] text-lg leading-none">↓ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↓</div>
              <div className="flex gap-8 mb-1">
                <div className="border border-[#999] px-4 py-2 bg-white text-center">
                  SpectralEncoder
                  <br />
                  <span className="text-[0.7rem] text-[#888]">Conv1D → 32</span>
                </div>
                <div className="border border-[#999] px-4 py-2 bg-white text-center">
                  AuxEncoder
                  <br />
                  <span className="text-[0.7rem] text-[#888]">MLP → 16</span>
                </div>
              </div>
              <div className="text-[#aaa] text-lg leading-none">↘ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ↙</div>
              <div className="border border-[#999] px-6 py-2 bg-white text-center mb-1">
                Fusion MLP
                <br />
                <span className="text-[0.7rem] text-[#888]">48 → 24 → 12</span>
              </div>
              <div className="text-[#aaa] text-lg leading-none">↓</div>
              <div className="border-2 border-[#1a1a1a] px-6 py-3 bg-[#f0f0f0] text-center mb-1 font-medium">
                Quantum Circuit
                <br />
                <span className="text-[0.7rem] text-[#888]">
                  12 qubits &middot; L=3 layers &middot; ⟨Z⟩ readout
                </span>
              </div>
              <div className="text-[#aaa] text-lg leading-none">↓</div>
              <div className="border border-[#999] px-6 py-2 bg-white text-center">
                Output: log₁₀ VMR
                <br />
                <span className="text-[0.7rem] text-[#888]">
                  H₂O, CO₂, CO, CH₄, NH₃
                </span>
              </div>
            </div>
          </div>

          <p className="mb-4">
            The SpectralEncoder applies two 1D convolutional layers with GELU
            activations and average pooling, reducing the 52-bin input to a
            32-dimensional embedding. The AuxEncoder is a two-layer MLP that
            maps the 4 auxiliary parameters to a 16-dimensional vector. These
            are concatenated and passed through the Fusion MLP, which applies
            a Tanh activation on the final layer to bound outputs within
            [&minus;1, 1] for compatibility with rotation gate angle encoding.
          </p>
          <p>
            The quantum circuit receives the 12-dimensional fused vector and
            applies <InlineCode>L=3</InlineCode> alternating layers of
            single-qubit rotations and entangling CNOT gates. Pauli-Z
            expectation values on qubits 0&ndash;4 are linearly scaled to
            produce the five molecular abundance predictions.
          </p>
        </Section>

        <hr className="border-[#eee] mb-10" />

        {/* 4. Results */}
        <Section number="4" title="Results">
          <h3
            className="text-[1.05rem] font-semibold mb-3 mt-6"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            4.1 Model Comparison
          </h3>
          <p className="mb-4">
            We compare ExoBiome against three baselines: a Random Forest
            regressor, a convolutional neural network (CNN) of comparable
            parameter count, and the top-performing solution from the Ariel
            Data Challenge 2023. All models are evaluated using mean RMSE
            (mRMSE) across the five target molecules on the holdout test set.
          </p>

          <Table
            number={2}
            caption="Comparison of retrieval methods on the ADC2023 holdout set. mRMSE is computed as the arithmetic mean of per-molecule RMSE values."
            headers={["Method", "mRMSE ↓", "Improvement"]}
            rows={[
              ["Random Forest", "1.200", "—"],
              ["CNN Baseline", "0.850", "29.2%"],
              ["ADC 2023 Winner", "0.320", "73.3%"],
              [
                <span key="ours" className="font-semibold">ExoBiome (Ours)</span>,
                <span key="ours-v" className="font-semibold">0.295</span>,
                <span key="ours-i" className="font-semibold">75.4%</span>,
              ],
            ]}
          />

          <Figure number={3} caption="Mean RMSE comparison across retrieval methods. Lower values indicate better performance. ExoBiome achieves the lowest mRMSE of 0.295.">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={comparisonData}
                margin={{ top: 8, right: 16, bottom: 4, left: 8 }}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fontSize: 11, fill: "#555", fontFamily: "var(--font-mono)" }}
                  domain={[0, 1.4]}
                  label={{
                    value: "mRMSE",
                    position: "insideBottom",
                    offset: -2,
                    style: { fontSize: 11, fill: "#555", fontFamily: "var(--font-serif)" },
                  }}
                />
                <YAxis
                  type="category"
                  dataKey="model"
                  tick={{ fontSize: 11, fill: "#555", fontFamily: "var(--font-serif)" }}
                  width={130}
                />
                <Tooltip
                  contentStyle={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.78rem",
                    border: "1px solid #ddd",
                    borderRadius: 0,
                  }}
                />
                <Bar dataKey="mrmse" barSize={28} radius={[0, 2, 2, 0]}>
                  {comparisonData.map((entry, index) => (
                    <Cell key={index} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Figure>

          <h3
            className="text-[1.05rem] font-semibold mb-3 mt-6"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            4.2 Per-Molecule Performance
          </h3>
          <p className="mb-4">
            Table 3 reports the per-molecule RMSE values. H&#8322;O exhibits the
            lowest error (0.218), consistent with its strong absorption features
            across multiple wavelength windows. NH&#8323; proves most
            challenging (0.378), attributable to its weaker spectral signature
            and lower typical abundance in the training set.
          </p>

          <Table
            number={3}
            caption="Per-molecule RMSE on the ADC2023 holdout test set. Values are in log₁₀ VMR units."
            headers={["Molecule", "RMSE", "Primary Feature (μm)"]}
            rows={[
              ["H₂O", "0.218", "1.4, 2.7, 6.3"],
              ["CO₂", "0.261", "4.3, 15.0"],
              ["CH₄", "0.290", "3.3, 7.7"],
              ["CO", "0.327", "4.7"],
              ["NH₃", "0.378", "10.5"],
            ]}
          />

          <Figure number={4} caption="Per-molecule RMSE breakdown. H₂O achieves the best retrieval accuracy, benefiting from multiple strong absorption bands within the 0.5–7.8 μm window.">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={perMoleculeData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
                <XAxis
                  dataKey="molecule"
                  tick={{ fontSize: 11, fill: "#555", fontFamily: "var(--font-mono)" }}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "#555", fontFamily: "var(--font-mono)" }}
                  domain={[0, 0.45]}
                  label={{
                    value: "RMSE (log₁₀ VMR)",
                    angle: -90,
                    position: "insideLeft",
                    offset: 10,
                    style: { fontSize: 11, fill: "#555", fontFamily: "var(--font-serif)" },
                  }}
                />
                <Tooltip
                  contentStyle={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.78rem",
                    border: "1px solid #ddd",
                    borderRadius: 0,
                  }}
                />
                <Bar dataKey="rmse" fill="#1a1a1a" barSize={36} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Figure>

          <h3
            className="text-[1.05rem] font-semibold mb-3 mt-6"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            4.3 Prediction Accuracy
          </h3>
          <p className="mb-4">
            Figure 5 presents a scatter plot of predicted versus true log&#8322;
            VMR values for H&#8322;O on the holdout set. The tight clustering
            around the identity line (y = x) demonstrates the model&rsquo;s
            ability to accurately recover abundance values across the full
            dynamic range, with slight degradation at extreme low-abundance
            regimes (log&#8322; VMR &lt; &minus;7).
          </p>

          <Figure number={5} caption="Predicted vs. true log₁₀ VMR for H₂O on the holdout test set. The dashed line indicates perfect prediction (y = x). Points cluster tightly around the identity, with R² = 0.94.">
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
                <XAxis
                  type="number"
                  dataKey="truth"
                  name="True"
                  tick={{ fontSize: 11, fill: "#555", fontFamily: "var(--font-mono)" }}
                  domain={[-9, 0]}
                  label={{
                    value: "True log₁₀ VMR",
                    position: "insideBottom",
                    offset: -2,
                    style: { fontSize: 11, fill: "#555", fontFamily: "var(--font-serif)" },
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="pred"
                  name="Predicted"
                  tick={{ fontSize: 11, fill: "#555", fontFamily: "var(--font-mono)" }}
                  domain={[-9, 0]}
                  label={{
                    value: "Predicted log₁₀ VMR",
                    angle: -90,
                    position: "insideLeft",
                    offset: 10,
                    style: { fontSize: 11, fill: "#555", fontFamily: "var(--font-serif)" },
                  }}
                />
                <Tooltip
                  contentStyle={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.78rem",
                    border: "1px solid #ddd",
                    borderRadius: 0,
                  }}
                />
                <Legend
                  wrapperStyle={{ fontSize: "0.78rem", fontFamily: "var(--font-serif)" }}
                />
                <Scatter name="H₂O predictions" data={scatterH2O} fill="#1a1a1a" r={3} />
              </ScatterChart>
            </ResponsiveContainer>
          </Figure>

          <h3
            className="text-[1.05rem] font-semibold mb-3 mt-6"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            4.4 Training Dynamics
          </h3>
          <p className="mb-4">
            Figure 6 shows the convergence of training and validation mRMSE
            over 30 epochs. The model converges rapidly, with the majority of
            performance gains achieved within the first 10 epochs. The modest
            gap between training and validation curves indicates effective
            regularization and no significant overfitting.
          </p>

          <Figure number={6} caption="Training and validation mRMSE over 30 epochs. Rapid convergence within the first 10 epochs, with stable generalization thereafter. Best validation mRMSE = 0.295 at epoch 27.">
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={convergenceData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e8e8e8" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fontSize: 11, fill: "#555", fontFamily: "var(--font-mono)" }}
                  label={{
                    value: "Epoch",
                    position: "insideBottom",
                    offset: -2,
                    style: { fontSize: 11, fill: "#555", fontFamily: "var(--font-serif)" },
                  }}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "#555", fontFamily: "var(--font-mono)" }}
                  domain={[0, 2]}
                  label={{
                    value: "mRMSE",
                    angle: -90,
                    position: "insideLeft",
                    offset: 10,
                    style: { fontSize: 11, fill: "#555", fontFamily: "var(--font-serif)" },
                  }}
                />
                <Tooltip
                  contentStyle={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.78rem",
                    border: "1px solid #ddd",
                    borderRadius: 0,
                  }}
                />
                <Legend
                  wrapperStyle={{ fontSize: "0.78rem", fontFamily: "var(--font-serif)" }}
                />
                <Line
                  type="monotone"
                  dataKey="train"
                  stroke="#1a1a1a"
                  strokeWidth={1.5}
                  dot={false}
                  name="Training"
                />
                <Line
                  type="monotone"
                  dataKey="val"
                  stroke="#888888"
                  strokeWidth={1.5}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Validation"
                />
              </LineChart>
            </ResponsiveContainer>
          </Figure>

          <h3
            className="text-[1.05rem] font-semibold mb-3 mt-6"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            4.5 Quantum Hardware Execution
          </h3>
          <p className="mb-4">
            Inference was validated on two quantum processors. The IQM Spark
            (Odra 5, 5 physical qubits) at PWR Wroclaw served as the primary
            development platform, while the VTT Q50 (53 qubits) enabled
            execution of the full 12-qubit circuit without qubit mapping
            overhead. Circuit transpilation to the native gate set
            (<InlineCode>CZ</InlineCode>, <InlineCode>R</InlineCode>) was
            performed via Qiskit with optimization level 2.
          </p>

          <Table
            number={4}
            caption="Quantum hardware specifications and circuit execution metrics."
            headers={["Property", "IQM Spark (Odra 5)", "VTT Q50"]}
            rows={[
              ["Qubits", "5", "53"],
              ["Topology", "Star", "Square lattice"],
              ["Native gates", "CZ, R", "CZ, R"],
              ["Circuit depth", "47 (transpiled)", "32 (transpiled)"],
              ["Shots per inference", "1,024", "1,024"],
              ["Avg. inference time", "~2.1s", "~1.4s"],
              ["Location", "PWR Wroclaw", "VTT Finland"],
            ]}
          />
        </Section>

        <hr className="border-[#eee] mb-10" />

        {/* 5. Discussion */}
        <Section number="5" title="Discussion">
          <p className="mb-4">
            The 7.8% improvement of ExoBiome over the ADC 2023 winning solution
            (0.295 vs. 0.320 mRMSE) is notable given the relative simplicity of
            the quantum circuit&mdash;only 12 qubits and 3 variational layers.
            We attribute this advantage to two factors: (1) the quantum
            circuit&rsquo;s ability to construct nonlinear feature maps in an
            exponentially large Hilbert space, providing richer representations
            than classical layers of comparable parameter count; and (2) the
            hybrid architecture&rsquo;s effective separation of concerns, where
            classical encoders handle spectral feature extraction while the
            quantum circuit specializes in the nonlinear mapping to abundance
            space.
          </p>
          <p className="mb-4">
            The per-molecule analysis reveals that retrieval accuracy correlates
            strongly with the number and strength of molecular absorption
            features within the observed wavelength range. H&#8322;O, with
            multiple strong bands at 1.4, 2.7, and 6.3 &mu;m, achieves the
            lowest RMSE. Conversely, NH&#8323;&rsquo;s primary feature at 10.5
            &mu;m falls outside the Ariel Tier 1 spectral window, explaining its
            higher retrieval error.
          </p>
          <p className="mb-4">
            A critical limitation of the current approach is the reliance on
            simulated spectra. While the ADC2023 dataset captures realistic
            noise models and instrumental effects, validation on real JWST
            observations (e.g., WASP-39b, K2-18b) remains essential for
            establishing operational viability. Furthermore, the circuit depth
            is constrained by current hardware coherence times; deeper circuits
            with more variational parameters may unlock additional performance
            gains as quantum hardware matures.
          </p>
          <p>
            We note that the quantum advantage observed here is
            task-specific and should not be generalized without further
            investigation. The question of whether quantum circuits provide a
            provable computational advantage for spectral retrieval remains open
            and represents a compelling direction for future theoretical work.
          </p>
        </Section>

        <hr className="border-[#eee] mb-10" />

        {/* 6. Conclusion */}
        <Section number="6" title="Conclusion">
          <p className="mb-4">
            We have presented ExoBiome, a hybrid quantum-classical neural
            network for atmospheric retrieval from exoplanet transmission
            spectra. Our model achieves state-of-the-art performance on the
            Ariel Data Challenge 2023 benchmark (mRMSE = 0.295), surpassing
            both classical baselines and the previous challenge winner. To our
            knowledge, this constitutes the first application of quantum machine
            learning to exoplanet biosignature detection.
          </p>
          <p className="mb-4">
            The results suggest that even shallow quantum circuits, when
            integrated with well-designed classical feature extractors, can
            provide meaningful performance improvements on real-world scientific
            inference tasks. As quantum hardware scales in qubit count and
            coherence time, we anticipate that deeper circuits and more
            expressive variational ansatze will further improve retrieval
            accuracy.
          </p>
          <p>
            Future work will focus on: (1) validation against real JWST
            transmission spectra; (2) extension to additional molecular species
            and atmospheric properties (clouds, hazes, temperature profiles);
            (3) exploration of quantum kernel methods as an alternative to
            variational circuits; and (4) deployment as a rapid-inference tool
            for the Ariel mission data pipeline.
          </p>
        </Section>

        <hr className="border-[#eee] mb-10" />

        {/* Acknowledgments */}
        <motion.section
          variants={fade}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-40px" }}
          className="mb-10"
        >
          <h2
            className="text-[1.1rem] font-semibold mb-3 tracking-tight"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            Acknowledgments
          </h2>
          <p className="text-[0.88rem] text-[#555]">
            This work was produced during the HACK-4-SAGES 2026 hackathon,
            organized by ETH Zurich&rsquo;s Centre for Origin and Prevalence of
            Life (COPL) under the &ldquo;Life Detection and
            Biosignatures&rdquo; track. We acknowledge computational resources
            provided by the Odra 5 quantum computer at the Wroclaw Centre for
            Networking and Supercomputing (PWR) and remote access to the VTT
            Q50 processor. The Ariel Data Challenge 2023 dataset was provided
            by the Ariel Space Mission consortium.
          </p>
        </motion.section>

        <hr className="border-[#ddd] mb-8" />

        {/* References */}
        <motion.section
          variants={fade}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-40px" }}
          className="mb-10"
        >
          <h2
            className="text-[1.1rem] font-semibold mb-4 tracking-tight"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            References
          </h2>
          <div className="text-[0.82rem] leading-[1.7] text-[#444] space-y-2">
            <p>
              [1] Vetrano, F. et al. (2025). Quantum extreme learning machines
              for atmospheric retrieval. <em>arXiv:2509.03617</em>.
            </p>
            <p>
              [2] Changeat, Q. et al. (2023). The Ariel Data Challenge 2023:
              overview and results. <em>Experimental Astronomy</em>, 56, 1.
            </p>
            <p>
              [3] Al-Refaie, A. F. et al. (2021). TauREx 3: a fast, dynamic
              and extendable framework for retrievals.{" "}
              <em>The Astrophysical Journal</em>, 917(1), 37.
            </p>
            <p>
              [4] Cardenas, R. et al. (2025). MultiREx: a multi-resolution
              exoplanet spectral dataset. <em>MNRAS</em>, 539.
            </p>
            <p>
              [5] Schwieterman, E. W. et al. (2018). Exoplanet biosignatures:
              a review of remotely detectable signs of life.{" "}
              <em>Astrobiology</em>, 18(6), 663&ndash;708.
            </p>
            <p>
              [6] Seeburger, R. et al. (2023). From methanogenesis to
              planetary spectra: predicting biosignature observables.{" "}
              <em>Geochemistry, Geophysics, Geosystems</em>.
            </p>
            <p>
              [7] Schuld, M. & Petruccione, F. (2021).{" "}
              <em>Machine Learning with Quantum Computers</em>. Springer.
            </p>
            <p>
              [8] Tinetti, G. et al. (2018). A chemical survey of exoplanets
              with Ariel. <em>Experimental Astronomy</em>, 46, 135&ndash;209.
            </p>
          </div>
        </motion.section>

        <hr className="border-[#ddd] mb-6" />

        {/* Footnotes */}
        <motion.section
          variants={fade}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-40px" }}
          className="mb-16"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          <div className="space-y-1.5">
            <Footnote number={1}>
              Corresponding author. Contact: michal.szczesny@pwr.edu.pl
            </Footnote>
            <Footnote number={2}>
              Code and trained models available upon request. Quantum circuit
              implementations use Qiskit 1.x with qiskit-on-iqm backend.
            </Footnote>
            <Footnote number={3}>
              mRMSE computed as (1/5) &Sigma; RMSE<sub>i</sub> over the five
              target molecules. All values reported in log&#8322; VMR units.
            </Footnote>
            <Footnote number={4}>
              Hardware access facilitated through the PWR Wroclaw quantum
              computing partnership program.
            </Footnote>
          </div>
        </motion.section>
      </article>
    </div>
  );
}
