"use client";

import { useState } from "react";
import { Inter, JetBrains_Mono } from "next/font/google";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  Label,
} from "recharts";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

/* ─── Data ───────────────────────────────────────────────────────────── */

const moleculeResults = [
  { molecule: "H₂O", rmse: 0.218, color: "#2196F3" },
  { molecule: "CO₂", rmse: 0.261, color: "#4CAF50" },
  { molecule: "CH₄", rmse: 0.29, color: "#FF9800" },
  { molecule: "CO", rmse: 0.327, color: "#9C27B0" },
  { molecule: "NH₃", rmse: 0.378, color: "#F44336" },
];

const benchmarkData = [
  { model: "Random Forest", mrmse: 1.2, fill: "#E0E0E0" },
  { model: "CNN Baseline", mrmse: 0.85, fill: "#BDBDBD" },
  { model: "ADC 2023 Winner", mrmse: 0.32, fill: "#90CAF9" },
  { model: "ExoBiome (ours)", mrmse: 0.295, fill: "#2563EB" },
];

const techStack = [
  { label: "Python", color: "#3776AB" },
  { label: "PyTorch", color: "#EE4C2C" },
  { label: "Qiskit", color: "#6929C4" },
  { label: "qiskit-on-iqm", color: "#8B5CF6" },
  { label: "sQUlearn", color: "#7C3AED" },
  { label: "scikit-learn", color: "#F7931E" },
  { label: "TauREx 3", color: "#059669" },
  { label: "SpectRes", color: "#0891B2" },
  { label: "Jupyter", color: "#E76F00" },
  { label: "Recharts", color: "#22C55E" },
];

const moleculePills: { label: string; color: string }[] = [
  { label: "H₂O", color: "#2196F3" },
  { label: "CO₂", color: "#4CAF50" },
  { label: "CO", color: "#9C27B0" },
  { label: "CH₄", color: "#FF9800" },
  { label: "NH₃", color: "#F44336" },
];

/* ─── Notion-style primitives ────────────────────────────────────────── */

function Divider() {
  return <hr className="border-t border-gray-200 my-8" />;
}

function InlineCode({ children }: { children: React.ReactNode }) {
  return (
    <code
      className="px-1.5 py-0.5 rounded text-[0.85em] font-[family-name:var(--font-jetbrains)]"
      style={{ background: "#F1F1EF", color: "#EB5757" }}
    >
      {children}
    </code>
  );
}

function Callout({
  type,
  emoji,
  children,
}: {
  type: "blue" | "green" | "yellow" | "red";
  emoji?: string;
  children: React.ReactNode;
}) {
  const colors = {
    blue: { bg: "#E8F0FE", border: "#2563EB" },
    green: { bg: "#E6F4EA", border: "#16A34A" },
    yellow: { bg: "#FEF7E0", border: "#CA8A04" },
    red: { bg: "#FEE2E2", border: "#DC2626" },
  };
  const c = colors[type];
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.35 }}
      className="rounded-md px-5 py-4 my-4 text-[15px] leading-relaxed"
      style={{
        background: c.bg,
        borderLeft: `3px solid ${c.border}`,
      }}
    >
      <div className="flex gap-3 items-start">
        {emoji && <span className="text-xl mt-0.5 shrink-0">{emoji}</span>}
        <div className="flex-1 min-w-0">{children}</div>
      </div>
    </motion.div>
  );
}

function Toggle({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="my-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full text-left py-1.5 group cursor-pointer"
      >
        <motion.span
          animate={{ rotate: open ? 90 : 0 }}
          transition={{ duration: 0.15 }}
          className="text-gray-400 text-xs inline-block w-4 text-center"
        >
          ▶
        </motion.span>
        <span className="font-semibold text-[15px] text-gray-900 group-hover:text-gray-600 transition-colors">
          {title}
        </span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="pl-6 pb-2 text-[15px] leading-relaxed text-gray-700">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Quote({ children }: { children: React.ReactNode }) {
  return (
    <blockquote
      className="my-4 pl-4 py-1 text-[15px] italic text-gray-500"
      style={{ borderLeft: "3px solid #E0E0E0" }}
    >
      {children}
    </blockquote>
  );
}

function Pill({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium mr-2 mb-2"
      style={{
        background: `${color}18`,
        color: color,
        border: `1px solid ${color}30`,
      }}
    >
      {label}
    </span>
  );
}

function SectionHeading({
  children,
  id,
}: {
  children: React.ReactNode;
  id?: string;
}) {
  return (
    <motion.h2
      id={id}
      initial={{ opacity: 0, y: 6 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-30px" }}
      transition={{ duration: 0.3 }}
      className="text-[1.5rem] font-bold text-gray-900 mt-10 mb-3"
    >
      {children}
    </motion.h2>
  );
}

function SubHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-[1.15rem] font-semibold text-gray-800 mt-6 mb-2">
      {children}
    </h3>
  );
}

function Paragraph({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[15px] leading-[1.7] text-gray-700 my-3">{children}</p>
  );
}

function CodeBlock({ children }: { children: string }) {
  return (
    <motion.pre
      initial={{ opacity: 0, y: 6 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.3 }}
      className="my-4 p-4 rounded-lg text-[13px] leading-relaxed overflow-x-auto font-[family-name:var(--font-jetbrains)]"
      style={{ background: "#F7F6F3", color: "#37352F" }}
    >
      <code>{children}</code>
    </motion.pre>
  );
}

/* ─── Custom Tooltip ─────────────────────────────────────────────────── */

function BenchmarkTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { model: string; mrmse: number } }> }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm px-3 py-2 text-sm">
      <p className="font-medium text-gray-900">{d.model}</p>
      <p className="text-gray-500">
        mRMSE: <span className="font-semibold text-gray-800">{d.mrmse}</span>
      </p>
    </div>
  );
}

function MoleculeTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { molecule: string; rmse: number } }> }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm px-3 py-2 text-sm">
      <p className="font-medium text-gray-900">{d.molecule}</p>
      <p className="text-gray-500">
        RMSE: <span className="font-semibold text-gray-800">{d.rmse}</span>
      </p>
    </div>
  );
}

/* ─── Architecture Diagram (ASCII) ───────────────────────────────────── */

const architectureASCII = `┌─────────────────────────────────────────────────────────────┐
│                    ExoBiome Architecture                     │
└─────────────────────────────────────────────────────────────┘

  Transmission Spectrum (52 wavelength bins)
        │
        ▼
  ┌─────────────────┐     Auxiliary Features
  │ SpectralEncoder  │     (star temp, radius,
  │  Conv1D → GELU   │     planet mass, etc.)
  │  + Residuals     │            │
  └────────┬────────┘            │
           │                     ▼
           │            ┌────────────────┐
           │            │   AuxEncoder   │
           │            │   Linear+GELU  │
           │            └───────┬────────┘
           │                    │
           └──────┬─────────────┘
                  ▼
          ┌───────────────┐
          │  Fusion Layer  │
          │  Concat + MLP  │
          └───────┬───────┘
                  │
                  ▼
    ┌───────────────────────────┐
    │    Quantum Circuit Layer   │
    │    12 qubits (IQM Spark)   │
    │    RY encoding + CZ gates  │
    │    PauliZ measurements     │
    └─────────────┬─────────────┘
                  │
                  ▼
          ┌───────────────┐
          │  Output Head   │
          │  Linear → 5    │
          └───────┬───────┘
                  │
                  ▼
    log₁₀ VMR predictions for:
    H₂O, CO₂, CO, CH₄, NH₃`;

/* ─── Page ───────────────────────────────────────────────────────────── */

export default function ExoBiomePage() {
  return (
    <div
      className={`${inter.variable} ${jetbrains.variable} min-h-screen`}
      style={{
        background: "#FFFFFF",
        fontFamily: "var(--font-inter), system-ui, sans-serif",
      }}
    >
      {/* Cover gradient strip */}
      <div
        className="w-full h-52 relative"
        style={{
          background:
            "linear-gradient(135deg, #DBEAFE 0%, #EDE9FE 35%, #F0FDFA 65%, #FFFFFF 100%)",
        }}
      >
        <div className="absolute inset-0 opacity-[0.04]"
          style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, #000 1px, transparent 0)`,
            backgroundSize: "24px 24px",
          }}
        />
      </div>

      {/* Page icon + title area */}
      <div className="max-w-[800px] mx-auto px-6 -mt-10 relative">
        {/* Emoji icon */}
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="w-[72px] h-[72px] rounded-2xl flex items-center justify-center text-[40px] mb-4 shadow-sm border border-gray-100"
          style={{ background: "#FFFFFF" }}
        >
          🔬
        </motion.div>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-[2.5rem] font-bold text-gray-900 leading-tight tracking-[-0.02em]"
        >
          ExoBiome
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
          className="text-[1.1rem] text-gray-500 mt-2 mb-1 leading-relaxed"
        >
          Quantum-Enhanced Biosignature Detection from Exoplanet Transmission
          Spectra
        </motion.p>

        {/* Meta row */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="flex items-center gap-4 text-sm text-gray-400 mt-3 mb-8"
        >
          <span>HACK-4-SAGES 2026</span>
          <span>·</span>
          <span>ETH Zurich (COPL)</span>
          <span>·</span>
          <span>Life Detection &amp; Biosignatures</span>
        </motion.div>

        <Divider />

        {/* ── Quick Summary ── */}
        <Callout type="blue" emoji="💡">
          <span className="font-semibold">Quick summary</span> — ExoBiome is a
          quantum-classical hybrid neural network that predicts molecular
          abundances (log₁₀ VMR) from exoplanet transmission spectra. Trained on
          106k+ synthetic spectra and evaluated against the Ariel Data Challenge
          2023 format, it achieves a{" "}
          <strong>mean RMSE of 0.295</strong> across five target molecules —
          outperforming the ADC 2023 winning solution (~0.32). The quantum
          circuit layer runs on 12 qubits targeting IQM Spark hardware.
        </Callout>

        <Divider />

        {/* ── Background ── */}
        <SectionHeading id="background">Background</SectionHeading>

        <Paragraph>
          The search for life beyond Earth hinges on our ability to detect
          molecular signatures in exoplanet atmospheres. Transmission
          spectroscopy — measuring starlight filtered through a planet&apos;s
          atmosphere during transit — reveals absorption features tied to
          specific molecules. The challenge: extracting precise molecular
          abundances from noisy, low-resolution spectra.
        </Paragraph>

        <Toggle title="Why quantum machine learning?">
          <Paragraph>
            Quantum computing offers potential advantages for specific ML tasks
            through properties like superposition and entanglement. Quantum
            Extreme Learning Machines (QELMs), as demonstrated by{" "}
            <InlineCode>Vetrano et al. 2025</InlineCode>, can serve as
            expressive feature maps with trainable parameters only in the
            classical readout layer. This simplifies training while leveraging
            quantum Hilbert space dimensionality.
          </Paragraph>
          <Paragraph>
            ExoBiome adapts this concept with a hybrid approach: classical
            encoders compress the input, and a parameterized quantum circuit
            acts as the penultimate feature transformation before the final
            prediction head.
          </Paragraph>
        </Toggle>

        <Toggle title="The Ariel Data Challenge 2023">
          <Paragraph>
            The Ariel Data Challenge (ADC) 2023 established a standardized
            benchmark for atmospheric retrieval from synthetic transit spectra.
            The task: given a transmission spectrum with 52 wavelength bins plus
            auxiliary planetary/stellar parameters, predict the log₁₀ volume
            mixing ratios (VMR) for five molecules:{" "}
            {moleculePills.map((m) => (
              <Pill key={m.label} label={m.label} color={m.color} />
            ))}
          </Paragraph>
          <Paragraph>
            We adopt this exact format as our evaluation protocol, training on
            the ABC Database (106k spectra) and validating against ADC ground
            truth distributions.
          </Paragraph>
        </Toggle>

        <Toggle title="Dataset & preprocessing">
          <Paragraph>
            Primary training data comes from the{" "}
            <strong>ABC Database</strong> (Zenodo 6770103): 106,000 synthetic
            transmission spectra generated with TauREx 3, perfectly aligned with
            the ADC 2023 format. Spectra are normalized per-sample and
            augmented with Gaussian noise injection. Auxiliary features
            (stellar temperature, stellar radius, planet mass, planet radius,
            planet temperature) are standardized with{" "}
            <InlineCode>RobustScaler</InlineCode>.
          </Paragraph>
        </Toggle>

        <Divider />

        {/* ── Architecture ── */}
        <SectionHeading id="architecture">Architecture</SectionHeading>

        <Paragraph>
          The model follows a modular encode → fuse → quantize → predict
          pipeline. Each component is designed to be lightweight — the total
          classical parameter count is under 50k — with the quantum circuit
          providing an expressive nonlinear feature transformation in 12-qubit
          Hilbert space.
        </Paragraph>

        <CodeBlock>{architectureASCII}</CodeBlock>

        <Toggle title="SpectralEncoder details" defaultOpen>
          <Paragraph>
            The spectral branch uses stacked 1D convolutions with{" "}
            <InlineCode>GELU</InlineCode> activations and residual connections.
            Input shape: <InlineCode>(batch, 1, 52)</InlineCode>. The encoder
            compresses 52 spectral bins into a 32-dimensional embedding through
            three convolutional blocks with kernel sizes [5, 3, 3] and channel
            progression [1 → 16 → 32 → 32], followed by adaptive average
            pooling and a linear projection.
          </Paragraph>
        </Toggle>

        <Toggle title="AuxEncoder details">
          <Paragraph>
            Auxiliary planetary/stellar features (5 inputs) pass through a
            two-layer MLP: <InlineCode>5 → 16 → 16</InlineCode> with{" "}
            <InlineCode>GELU</InlineCode> and{" "}
            <InlineCode>BatchNorm</InlineCode>. This branch captures
            non-spectral context that constrains the retrieval (e.g., a
            hot Jupiter vs. a temperate rocky planet).
          </Paragraph>
        </Toggle>

        <Toggle title="Quantum circuit layer">
          <Paragraph>
            The fused 48-dimensional classical embedding (32 + 16) is projected
            to 12 features, each encoding the rotation angle of an{" "}
            <InlineCode>RY</InlineCode> gate on one of 12 qubits. The circuit
            applies two layers of <InlineCode>CZ</InlineCode> entangling gates
            in a circular topology, followed by{" "}
            <InlineCode>PauliZ</InlineCode> expectation value measurements on
            all qubits. The 12 measurement outcomes feed the final linear
            output head.
          </Paragraph>
          <Callout type="yellow" emoji="⚠️">
            The quantum circuit is simulated during training (statevector
            backend). Inference can target real hardware via{" "}
            <InlineCode>qiskit-on-iqm</InlineCode> on the Odra 5 (IQM Spark,
            5 qubits) or VTT Q50 (53 qubits). The 12-qubit design fits within
            VTT Q50 natively; for Odra 5, circuit cutting splits the workload.
          </Callout>
        </Toggle>

        <Divider />

        {/* ── Results ── */}
        <SectionHeading id="results">Results</SectionHeading>

        <Callout type="green" emoji="🏆">
          <span className="font-semibold">
            ExoBiome achieves mRMSE = 0.295
          </span>{" "}
          on the holdout test set (ADC 2023 format), surpassing the ADC 2023
          competition winner (~0.32) and significantly outperforming classical
          baselines. This represents the first successful application of quantum
          machine learning to biosignature detection.
        </Callout>

        <SubHeading>Model Comparison</SubHeading>
        <Paragraph>
          Mean RMSE across all five target molecules (lower is better). The
          ExoBiome hybrid model outperforms both classical baselines and the
          previous best competition result.
        </Paragraph>

        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="my-6 -mx-2"
        >
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={benchmarkData}
              layout="vertical"
              margin={{ top: 8, right: 40, bottom: 8, left: 8 }}
              barSize={32}
            >
              <CartesianGrid
                horizontal={false}
                strokeDasharray="3 3"
                stroke="#E5E7EB"
              />
              <XAxis
                type="number"
                domain={[0, 1.4]}
                tick={{ fontSize: 12, fill: "#9CA3AF" }}
                axisLine={{ stroke: "#E5E7EB" }}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="model"
                width={140}
                tick={{ fontSize: 13, fill: "#374151" }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                content={<BenchmarkTooltip />}
                cursor={{ fill: "#F9FAFB" }}
              />
              <Bar dataKey="mrmse" radius={[0, 4, 4, 0]}>
                {benchmarkData.map((entry, index) => (
                  <Cell key={index} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        <Divider />

        {/* ── Per-molecule ── */}
        <SectionHeading id="per-molecule">
          Per-Molecule Performance
        </SectionHeading>

        <Paragraph>
          RMSE breakdown by molecule on the holdout set. Water (H₂O) is
          predicted most accurately due to its strong, broad absorption features.
          Ammonia (NH₃) is hardest, consistent with its weaker spectral
          signature in the NIR.
        </Paragraph>

        {/* Chart */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="my-6 -mx-2"
        >
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={moleculeResults}
              margin={{ top: 8, right: 20, bottom: 24, left: 0 }}
              barSize={52}
            >
              <CartesianGrid
                vertical={false}
                strokeDasharray="3 3"
                stroke="#E5E7EB"
              />
              <XAxis
                dataKey="molecule"
                tick={{ fontSize: 13, fill: "#374151" }}
                axisLine={{ stroke: "#E5E7EB" }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 0.5]}
                tick={{ fontSize: 12, fill: "#9CA3AF" }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                content={<MoleculeTooltip />}
                cursor={{ fill: "#F9FAFB" }}
              />
              <ReferenceLine
                y={0.32}
                stroke="#94A3B8"
                strokeDasharray="6 4"
                strokeWidth={1.5}
              >
                <Label
                  value="ADC Winner (0.32)"
                  position="right"
                  fill="#94A3B8"
                  fontSize={11}
                />
              </ReferenceLine>
              <Bar dataKey="rmse" radius={[4, 4, 0, 0]}>
                {moleculeResults.map((entry, index) => (
                  <Cell key={index} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Table */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.3 }}
          className="my-6 overflow-x-auto"
        >
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 text-xs uppercase tracking-wider">
                <th className="py-3 px-4 font-medium">Molecule</th>
                <th className="py-3 px-4 font-medium">Formula</th>
                <th className="py-3 px-4 font-medium text-right">RMSE</th>
                <th className="py-3 px-4 font-medium text-right">
                  vs ADC Winner
                </th>
                <th className="py-3 px-4 font-medium">Biosignature Role</th>
              </tr>
            </thead>
            <tbody>
              {[
                {
                  name: "Water",
                  formula: "H₂O",
                  rmse: 0.218,
                  role: "Habitability indicator",
                },
                {
                  name: "Carbon Dioxide",
                  formula: "CO₂",
                  rmse: 0.261,
                  role: "Atmospheric composition",
                },
                {
                  name: "Methane",
                  formula: "CH₄",
                  rmse: 0.29,
                  role: "Biological metabolism",
                },
                {
                  name: "Carbon Monoxide",
                  formula: "CO",
                  rmse: 0.327,
                  role: "Disequilibrium marker",
                },
                {
                  name: "Ammonia",
                  formula: "NH₃",
                  rmse: 0.378,
                  role: "Biogenic gas (cold atm.)",
                },
              ].map((row, i) => (
                <tr
                  key={row.formula}
                  className="border-t border-gray-100"
                  style={{
                    background: i % 2 === 0 ? "#FFFFFF" : "#FAFAFA",
                  }}
                >
                  <td className="py-3 px-4 text-gray-800 font-medium">
                    {row.name}
                  </td>
                  <td className="py-3 px-4">
                    <Pill
                      label={row.formula}
                      color={
                        moleculePills.find((m) => m.label === row.formula)
                          ?.color || "#666"
                      }
                    />
                  </td>
                  <td className="py-3 px-4 text-right font-[family-name:var(--font-jetbrains)] text-gray-900">
                    {row.rmse.toFixed(3)}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <span
                      className="text-xs font-medium px-2 py-0.5 rounded-full"
                      style={{
                        background:
                          row.rmse < 0.32 ? "#DCFCE7" : "#FEF3C7",
                        color:
                          row.rmse < 0.32 ? "#166534" : "#92400E",
                      }}
                    >
                      {row.rmse < 0.32
                        ? `↓ ${((0.32 - row.rmse) * 100).toFixed(1)}%`
                        : `↑ ${((row.rmse - 0.32) * 100).toFixed(1)}%`}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-gray-500 text-[13px]">
                    {row.role}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>

        <Quote>
          The co-detection of CH₄ and CO₂ in the absence of CO is considered a
          strong biosignature — the so-called thermodynamic disequilibrium
          argument. ExoBiome&apos;s ability to resolve all three simultaneously
          enables direct disequilibrium assessment from a single spectrum.
        </Quote>

        <Divider />

        {/* ── Key Innovations ── */}
        <SectionHeading id="innovations">Key Innovations</SectionHeading>

        <div className="space-y-2 my-4">
          <Toggle title="First quantum ML applied to biosignature detection" defaultOpen>
            <Paragraph>
              While <InlineCode>Vetrano et al. 2025</InlineCode> demonstrated
              QELMs for atmospheric retrieval (predicting molecular abundances),
              their work did not target biosignature-specific molecules or frame
              the task in terms of life detection. ExoBiome is the first to
              apply quantum machine learning specifically to biosignature
              classification and quantification, connecting the
              biosphere → atmosphere → spectrum → inference chain.
            </Paragraph>
          </Toggle>

          <Toggle title="Hybrid architecture outperforms pure-classical approaches">
            <Paragraph>
              The quantum circuit layer acts as a highly expressive nonlinear
              feature map in 2¹² = 4096 dimensional Hilbert space. Combined
              with lightweight classical encoders, this achieves state-of-the-art
              results with fewer total parameters than competing deep learning
              models. The architecture is specifically designed for near-term
              quantum hardware with limited qubit counts and gate fidelities.
            </Paragraph>
          </Toggle>

          <Toggle title="End-to-end pipeline from spectrum to biosignature assessment">
            <Paragraph>
              Previous works treat retrieval (spectrum → abundances) and
              biosignature assessment (abundances → life detection) as separate
              steps. ExoBiome unifies these by predicting all five
              biosignature-relevant molecules jointly, enabling direct
              thermodynamic disequilibrium calculation from raw spectra without
              intermediate Bayesian retrieval.
            </Paragraph>
          </Toggle>
        </div>

        <Divider />

        {/* ── Quantum Hardware ── */}
        <SectionHeading id="hardware">Quantum Hardware</SectionHeading>

        <Paragraph>
          ExoBiome targets real quantum hardware for inference, demonstrating
          practical quantum advantage potential in astrophysics applications.
        </Paragraph>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.3 }}
          className="my-6 overflow-x-auto"
        >
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 text-xs uppercase tracking-wider">
                <th className="py-3 px-4 font-medium">System</th>
                <th className="py-3 px-4 font-medium">Qubits</th>
                <th className="py-3 px-4 font-medium">Chip</th>
                <th className="py-3 px-4 font-medium">Access</th>
                <th className="py-3 px-4 font-medium">SDK</th>
              </tr>
            </thead>
            <tbody>
              {[
                {
                  name: "Odra 5",
                  qubits: "5",
                  chip: "IQM Spark",
                  access: "PWR Wroclaw (on-site)",
                  sdk: "qiskit-on-iqm",
                },
                {
                  name: "VTT Q50",
                  qubits: "53",
                  chip: "IQM (custom)",
                  access: "Finland (remote via PWR)",
                  sdk: "qiskit-on-iqm",
                },
              ].map((row, i) => (
                <tr
                  key={row.name}
                  className="border-t border-gray-100"
                  style={{
                    background: i % 2 === 0 ? "#FFFFFF" : "#FAFAFA",
                  }}
                >
                  <td className="py-3 px-4 text-gray-800 font-medium">
                    {row.name}
                  </td>
                  <td className="py-3 px-4 font-[family-name:var(--font-jetbrains)] text-gray-900">
                    {row.qubits}
                  </td>
                  <td className="py-3 px-4 text-gray-600">{row.chip}</td>
                  <td className="py-3 px-4 text-gray-500 text-[13px]">
                    {row.access}
                  </td>
                  <td className="py-3 px-4">
                    <InlineCode>{row.sdk}</InlineCode>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>

        <Divider />

        {/* ── Tech Stack ── */}
        <SectionHeading id="stack">Tech Stack</SectionHeading>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.3 }}
          className="my-4 flex flex-wrap"
        >
          {techStack.map((item) => (
            <Pill key={item.label} label={item.label} color={item.color} />
          ))}
        </motion.div>

        <Divider />

        {/* ── References ── */}
        <SectionHeading id="references">References</SectionHeading>

        <div className="my-4 space-y-3 text-[14px] text-gray-600">
          <div className="flex gap-3">
            <span className="text-gray-400 shrink-0 font-[family-name:var(--font-jetbrains)] text-xs mt-0.5">
              [1]
            </span>
            <span>
              Vetrano, F. et al. (2025). Quantum Extreme Learning Machine for
              Atmospheric Retrieval.{" "}
              <span className="italic">arXiv:2509.03617</span>
            </span>
          </div>
          <div className="flex gap-3">
            <span className="text-gray-400 shrink-0 font-[family-name:var(--font-jetbrains)] text-xs mt-0.5">
              [2]
            </span>
            <span>
              Cardenas, R. et al. (2025). MultiREx: Multi-planet Retrieval
              Exercise Dataset.{" "}
              <span className="italic">MNRAS 539</span>
            </span>
          </div>
          <div className="flex gap-3">
            <span className="text-gray-400 shrink-0 font-[family-name:var(--font-jetbrains)] text-xs mt-0.5">
              [3]
            </span>
            <span>
              Schwieterman, E. W. et al. (2018). Exoplanet Biosignatures: A
              Review of Remotely Detectable Signs of Life.{" "}
              <span className="italic">Astrobiology 18(6)</span>
            </span>
          </div>
          <div className="flex gap-3">
            <span className="text-gray-400 shrink-0 font-[family-name:var(--font-jetbrains)] text-xs mt-0.5">
              [4]
            </span>
            <span>
              Seeburger, R. et al. (2023). From Methanogenesis to Planetary
              Spectra: Predicting Observable Biosignatures.{" "}
              <span className="italic">ApJ</span>
            </span>
          </div>
          <div className="flex gap-3">
            <span className="text-gray-400 shrink-0 font-[family-name:var(--font-jetbrains)] text-xs mt-0.5">
              [5]
            </span>
            <span>
              Ariel Data Challenge 2023. European Space Agency.{" "}
              <span className="italic">ariel-datachallenge.space</span>
            </span>
          </div>
        </div>

        <Divider />

        {/* ── Footer ── */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="text-center py-12 text-sm text-gray-400"
        >
          <p>
            HACK-4-SAGES 2026 · ETH Zurich · Origins Federation
          </p>
          <p className="mt-1 text-xs text-gray-300">
            Category: Life Detection and Biosignatures
          </p>
        </motion.div>
      </div>
    </div>
  );
}
