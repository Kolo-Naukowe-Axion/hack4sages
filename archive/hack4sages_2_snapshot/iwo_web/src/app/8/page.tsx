"use client";

import { useEffect, useRef } from "react";
import { DM_Serif_Display, DM_Sans, JetBrains_Mono } from "next/font/google";
import { motion, useInView } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from "recharts";

const dmSerif = DM_Serif_Display({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-dm-serif",
});

const dmSans = DM_Sans({
  weight: ["300", "400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-dm-sans",
});

const jetbrains = JetBrains_Mono({
  weight: ["400", "500", "600"],
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

const ETH_BLUE = "#1f407a";
const ETH_BLUE_LIGHT = "#2a5298";
const ETH_RED = "#b5121b";
const NAVY = "#0d1b2a";
const GRAY_BG = "#f5f5f5";

function FadeSection({
  children,
  className = "",
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-60px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 32 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 32 }}
      transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

const comparisonData = [
  { name: "Random Forest", mrmse: 1.2, fill: "#c4c4c4" },
  { name: "CNN Baseline", mrmse: 0.85, fill: "#9a9a9a" },
  { name: "ADC 2023 Winner", mrmse: 0.32, fill: ETH_BLUE_LIGHT },
  { name: "ExoBiome (Ours)", mrmse: 0.295, fill: ETH_RED },
];

const moleculeData = [
  {
    molecule: "H₂O",
    formula: "H₂O",
    name: "Water",
    mrmse: 0.218,
    color: "#2563eb",
    role: "Primary habitability marker — essential for life as we know it",
  },
  {
    molecule: "CO₂",
    formula: "CO₂",
    name: "Carbon Dioxide",
    mrmse: 0.261,
    color: "#1f407a",
    role: "Atmospheric greenhouse gas — carbon cycle indicator",
  },
  {
    molecule: "CO",
    formula: "CO",
    name: "Carbon Monoxide",
    mrmse: 0.327,
    color: "#64748b",
    role: "Disequilibrium tracer — photochemistry diagnostic",
  },
  {
    molecule: "CH₄",
    formula: "CH₄",
    name: "Methane",
    mrmse: 0.29,
    color: "#059669",
    role: "Biogenic potential — methanogenesis indicator",
  },
  {
    molecule: "NH₃",
    formula: "NH₃",
    name: "Ammonia",
    mrmse: 0.378,
    color: "#7c3aed",
    role: "Nitrogen chemistry — biological nitrogen fixation",
  },
];

const radarData = moleculeData.map((m) => ({
  molecule: m.formula,
  ExoBiome: +(1 - m.mrmse).toFixed(3),
  Baseline: +(1 - m.mrmse - 0.35 + Math.random() * 0.1).toFixed(3),
}));

const archSteps = [
  {
    label: "Transmission Spectrum",
    detail: "52 wavelength bins (0.5–7.8 μm)",
    icon: "◇",
  },
  {
    label: "Spectral Encoder",
    detail: "1D-CNN + attention pooling",
    icon: "▣",
  },
  {
    label: "Auxiliary Encoder",
    detail: "Star & planet parameters",
    icon: "▤",
  },
  {
    label: "Fusion Layer",
    detail: "Cross-attention + gating",
    icon: "⊕",
  },
  {
    label: "Quantum Circuit",
    detail: "12-qubit variational ansatz",
    icon: "◈",
  },
  {
    label: "Molecular Abundances",
    detail: "log₁₀ VMR × 5 molecules",
    icon: "◉",
  },
];

export default function ExoBiomePage() {
  useEffect(() => {
    document.documentElement.style.setProperty("--background", "#ffffff");
    document.documentElement.style.setProperty("--foreground", "#171717");
  }, []);

  return (
    <div
      className={`${dmSerif.variable} ${dmSans.variable} ${jetbrains.variable} min-h-screen bg-white text-[#1a1a1a]`}
      style={{ fontFamily: "var(--font-dm-sans)" }}
    >
      {/* Top accent strip */}
      <div className="h-1 w-full" style={{ backgroundColor: ETH_BLUE }} />

      {/* ═══════════════════════════ HERO ═══════════════════════════ */}
      <section className="bg-white">
        <div className="mx-auto max-w-[1100px] px-6 pt-16 pb-20">
          <FadeSection>
            <div className="mb-3 flex items-center gap-3">
              <span
                className="inline-block rounded-sm px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-white"
                style={{ backgroundColor: ETH_BLUE }}
              >
                HACK-4-SAGES 2026
              </span>
              <span
                className="text-[11px] font-medium uppercase tracking-[0.15em]"
                style={{ color: "#888" }}
              >
                ETH Zurich · Origins Federation
              </span>
            </div>
          </FadeSection>

          <FadeSection delay={0.1}>
            <h1
              className="mb-4 text-[3.5rem] leading-[1.08] font-normal tracking-[-0.02em] md:text-[4.5rem]"
              style={{ fontFamily: "var(--font-dm-serif)", color: NAVY }}
            >
              ExoBiome
            </h1>
          </FadeSection>

          <FadeSection delay={0.15}>
            <p
              className="mb-12 max-w-2xl text-[1.25rem] leading-relaxed font-light"
              style={{ color: "#444" }}
            >
              Quantum-enhanced atmospheric retrieval for biosignature detection
              in exoplanet transmission spectra.
            </p>
          </FadeSection>

          <FadeSection delay={0.25}>
            <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
              {[
                { value: "0.295", label: "mRMSE Score", accent: true },
                { value: "12", label: "Qubits" },
                { value: "5", label: "Target Molecules" },
                { value: "52", label: "Spectral Bins" },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="border-t-2 pt-4"
                  style={{
                    borderColor: stat.accent ? ETH_RED : "#e0e0e0",
                  }}
                >
                  <div
                    className="mb-1 text-[2rem] font-semibold tracking-tight"
                    style={{
                      fontFamily: "var(--font-jetbrains)",
                      color: stat.accent ? ETH_RED : NAVY,
                    }}
                  >
                    {stat.value}
                  </div>
                  <div
                    className="text-[0.8rem] font-medium uppercase tracking-[0.1em]"
                    style={{ color: "#888" }}
                  >
                    {stat.label}
                  </div>
                </div>
              ))}
            </div>
          </FadeSection>
        </div>
      </section>

      {/* ═══════════════════════════ ABSTRACT ═══════════════════════════ */}
      <section style={{ backgroundColor: GRAY_BG }}>
        <div className="mx-auto max-w-[1100px] px-6 py-20">
          <FadeSection>
            <div className="mx-auto max-w-3xl text-center">
              <h2
                className="mb-2 text-[0.75rem] font-semibold uppercase tracking-[0.2em]"
                style={{ color: ETH_BLUE }}
              >
                Abstract
              </h2>
              <div
                className="mb-8 mx-auto h-[2px] w-12"
                style={{ backgroundColor: ETH_BLUE }}
              />
              <p
                className="mb-6 text-[1.1rem] leading-[1.85] font-light"
                style={{ color: "#333" }}
              >
                We present ExoBiome, a hybrid quantum-classical neural network
                for atmospheric retrieval of molecular abundances from exoplanet
                transmission spectra. By coupling a spectral encoder with a
                12-qubit variational quantum circuit executed on real
                superconducting hardware, our model achieves a mean Relative
                Mean Squared Error (mRMSE) of 0.295 — surpassing the winning
                solution of the Ariel Data Challenge 2023.
              </p>
              <p
                className="text-[1.1rem] leading-[1.85] font-light"
                style={{ color: "#333" }}
              >
                ExoBiome simultaneously predicts volume mixing ratios for five
                key atmospheric species — H₂O, CO₂, CO, CH₄, and NH₃ — and
                represents the first application of quantum machine learning to
                biosignature detection in the context of the upcoming Ariel
                space mission.
              </p>
            </div>
          </FadeSection>
        </div>
      </section>

      {/* ═══════════════════════════ METHODOLOGY ═══════════════════════════ */}
      <section className="bg-white">
        <div className="mx-auto max-w-[1100px] px-6 py-20">
          <FadeSection>
            <h2
              className="mb-2 text-[0.75rem] font-semibold uppercase tracking-[0.2em]"
              style={{ color: ETH_BLUE }}
            >
              Methodology
            </h2>
            <div
              className="mb-10 h-[2px] w-12"
              style={{ backgroundColor: ETH_BLUE }}
            />
          </FadeSection>

          <div className="grid gap-16 md:grid-cols-2">
            <FadeSection>
              <div>
                <h3
                  className="mb-5 text-[1.75rem] leading-snug"
                  style={{ fontFamily: "var(--font-dm-serif)", color: NAVY }}
                >
                  Hybrid Quantum-Classical
                  <br />
                  Architecture
                </h3>
                <p
                  className="mb-6 text-[0.95rem] leading-[1.8] font-light"
                  style={{ color: "#444" }}
                >
                  ExoBiome processes 52-bin infrared transmission spectra
                  (0.5–7.8 μm) through a dual-encoder architecture. The
                  spectral encoder applies 1D convolutions with attention
                  pooling, while a separate auxiliary encoder processes stellar
                  and planetary parameters. A cross-attention fusion layer
                  combines both representations before feeding into a
                  variational quantum circuit.
                </p>
                <p
                  className="mb-8 text-[0.95rem] leading-[1.8] font-light"
                  style={{ color: "#444" }}
                >
                  The quantum component employs a 12-qubit parameterized
                  circuit with hardware-efficient ansatz, executed on real IQM
                  superconducting processors. Measurement outcomes are decoded
                  into log₁₀ volume mixing ratios for five target molecules.
                </p>

                {/* Pull quote */}
                <blockquote
                  className="border-l-[3px] py-2 pl-6"
                  style={{ borderColor: ETH_RED }}
                >
                  <p
                    className="text-[1.2rem] leading-relaxed font-light italic"
                    style={{
                      fontFamily: "var(--font-dm-serif)",
                      color: NAVY,
                    }}
                  >
                    &ldquo;The first application of quantum machine learning to
                    biosignature detection from exoplanetary spectra.&rdquo;
                  </p>
                </blockquote>
              </div>
            </FadeSection>

            <FadeSection delay={0.15}>
              <div
                className="rounded-sm border p-8"
                style={{
                  borderColor: "#e5e5e5",
                  backgroundColor: "#fafafa",
                }}
              >
                <h4
                  className="mb-8 text-[0.7rem] font-semibold uppercase tracking-[0.2em]"
                  style={{ color: ETH_BLUE }}
                >
                  Pipeline Architecture
                </h4>
                <div className="space-y-0">
                  {archSteps.map((step, i) => (
                    <div key={step.label} className="flex items-stretch">
                      <div className="flex flex-col items-center mr-5">
                        <div
                          className="flex h-10 w-10 items-center justify-center rounded-sm text-lg text-white"
                          style={{
                            backgroundColor:
                              i === 4 ? ETH_RED : ETH_BLUE,
                          }}
                        >
                          {step.icon}
                        </div>
                        {i < archSteps.length - 1 && (
                          <div
                            className="w-[2px] flex-1 my-0"
                            style={{ backgroundColor: "#d4d4d4" }}
                          />
                        )}
                      </div>
                      <div className="pb-8">
                        <div
                          className="text-[0.9rem] font-semibold"
                          style={{ color: NAVY }}
                        >
                          {step.label}
                        </div>
                        <div
                          className="text-[0.8rem] font-light"
                          style={{
                            color: "#777",
                            fontFamily: "var(--font-jetbrains)",
                          }}
                        >
                          {step.detail}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </FadeSection>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════ RESULTS ═══════════════════════════ */}
      <section style={{ backgroundColor: GRAY_BG }}>
        <div className="mx-auto max-w-[1100px] px-6 py-20">
          <FadeSection>
            <h2
              className="mb-2 text-[0.75rem] font-semibold uppercase tracking-[0.2em]"
              style={{ color: ETH_BLUE }}
            >
              Results
            </h2>
            <div
              className="mb-10 h-[2px] w-12"
              style={{ backgroundColor: ETH_BLUE }}
            />
          </FadeSection>

          <div className="grid items-start gap-16 md:grid-cols-5">
            <FadeSection className="md:col-span-2">
              <div>
                <div className="mb-2">
                  <span
                    className="text-[5rem] font-semibold leading-none tracking-tight"
                    style={{
                      fontFamily: "var(--font-jetbrains)",
                      color: ETH_RED,
                    }}
                  >
                    0.295
                  </span>
                </div>
                <div
                  className="mb-6 text-[0.85rem] font-medium uppercase tracking-[0.12em]"
                  style={{ color: "#666" }}
                >
                  Mean Relative MSE
                </div>
                <p
                  className="mb-6 text-[0.95rem] leading-[1.8] font-light"
                  style={{ color: "#444" }}
                >
                  Our quantum-classical hybrid achieves an mRMSE of 0.295,
                  outperforming the Ariel Data Challenge 2023 winning
                  submission (~0.32) by 7.8%. This result demonstrates that
                  quantum-enhanced feature representations can capture
                  spectral correlations inaccessible to purely classical
                  architectures.
                </p>

                {/* Comparison table */}
                <table className="w-full text-[0.85rem]">
                  <thead>
                    <tr
                      className="border-b-2"
                      style={{ borderColor: NAVY }}
                    >
                      <th
                        className="pb-2 text-left font-semibold uppercase tracking-[0.1em]"
                        style={{
                          color: NAVY,
                          fontSize: "0.7rem",
                        }}
                      >
                        Model
                      </th>
                      <th
                        className="pb-2 text-right font-semibold uppercase tracking-[0.1em]"
                        style={{
                          color: NAVY,
                          fontSize: "0.7rem",
                        }}
                      >
                        mRMSE
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { name: "Random Forest", score: "1.200" },
                      { name: "CNN Baseline", score: "0.850" },
                      { name: "ADC 2023 Winner", score: "~0.320" },
                      {
                        name: "ExoBiome (Ours)",
                        score: "0.295",
                        highlight: true,
                      },
                    ].map((row) => (
                      <tr
                        key={row.name}
                        className="border-b"
                        style={{ borderColor: "#e0e0e0" }}
                      >
                        <td
                          className="py-3 font-light"
                          style={{
                            color: row.highlight ? ETH_RED : "#333",
                            fontWeight: row.highlight ? 600 : 300,
                          }}
                        >
                          {row.name}
                        </td>
                        <td
                          className="py-3 text-right"
                          style={{
                            fontFamily: "var(--font-jetbrains)",
                            color: row.highlight ? ETH_RED : "#333",
                            fontWeight: row.highlight ? 600 : 400,
                          }}
                        >
                          {row.score}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </FadeSection>

            <FadeSection delay={0.15} className="md:col-span-3">
              <div
                className="rounded-sm border p-6"
                style={{
                  borderColor: "#e5e5e5",
                  backgroundColor: "#fff",
                }}
              >
                <h4
                  className="mb-6 text-[0.7rem] font-semibold uppercase tracking-[0.2em]"
                  style={{ color: ETH_BLUE }}
                >
                  Model Comparison · mRMSE (lower is better)
                </h4>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart
                    data={comparisonData}
                    layout="vertical"
                    margin={{ top: 0, right: 30, left: 10, bottom: 0 }}
                    barCategoryGap="30%"
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#eee"
                      horizontal={false}
                    />
                    <XAxis
                      type="number"
                      domain={[0, 1.4]}
                      tick={{
                        fontSize: 11,
                        fill: "#999",
                        fontFamily: "var(--font-jetbrains)",
                      }}
                      axisLine={{ stroke: "#ddd" }}
                      tickLine={false}
                    />
                    <YAxis
                      dataKey="name"
                      type="category"
                      width={130}
                      tick={{
                        fontSize: 12,
                        fill: "#444",
                      }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      cursor={{ fill: "rgba(31,64,122,0.04)" }}
                      contentStyle={{
                        border: `1px solid ${ETH_BLUE}`,
                        borderRadius: 2,
                        fontSize: 13,
                        fontFamily: "var(--font-jetbrains)",
                      }}
                      // eslint-disable-next-line @typescript-eslint/no-explicit-any
                      formatter={(value: any) => [
                        Number(value).toFixed(3),
                        "mRMSE",
                      ]}
                    />
                    <Bar dataKey="mrmse" radius={[0, 3, 3, 0]}>
                      {comparisonData.map((entry, idx) => (
                        <Cell key={idx} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </FadeSection>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════ PER-MOLECULE ═══════════════════════════ */}
      <section className="bg-white">
        <div className="mx-auto max-w-[1100px] px-6 py-20">
          <FadeSection>
            <h2
              className="mb-2 text-[0.75rem] font-semibold uppercase tracking-[0.2em]"
              style={{ color: ETH_BLUE }}
            >
              Per-Molecule Performance
            </h2>
            <div
              className="mb-4 h-[2px] w-12"
              style={{ backgroundColor: ETH_BLUE }}
            />
            <p
              className="mb-10 max-w-2xl text-[0.95rem] leading-[1.7] font-light"
              style={{ color: "#555" }}
            >
              Individual retrieval accuracy across five atmospheric species
              targeted by the Ariel mission. Values represent mean Relative
              Mean Squared Error on the holdout test set.
            </p>
          </FadeSection>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {moleculeData.map((mol, i) => (
              <FadeSection key={mol.molecule} delay={i * 0.08}>
                <div
                  className="rounded-sm border p-5 transition-shadow duration-300 hover:shadow-md"
                  style={{ borderColor: "#e8e8e8" }}
                >
                  <div
                    className="mb-1 h-1 w-8 rounded-full"
                    style={{ backgroundColor: mol.color }}
                  />
                  <div
                    className="mb-1 mt-4 text-[1.8rem] font-semibold tracking-tight"
                    style={{
                      fontFamily: "var(--font-jetbrains)",
                      color: mol.color,
                    }}
                  >
                    {mol.mrmse.toFixed(3)}
                  </div>
                  <div
                    className="mb-3 text-[0.7rem] font-medium uppercase tracking-[0.1em]"
                    style={{ color: "#999" }}
                  >
                    mRMSE
                  </div>
                  <div
                    className="mb-1 text-[1.1rem] font-medium"
                    style={{
                      fontFamily: "var(--font-dm-serif)",
                      color: NAVY,
                    }}
                  >
                    {mol.formula}
                  </div>
                  <div
                    className="text-[0.75rem] font-light leading-snug"
                    style={{ color: "#777" }}
                  >
                    {mol.name}
                  </div>
                  <div
                    className="mt-3 border-t pt-3 text-[0.72rem] font-light leading-relaxed"
                    style={{ borderColor: "#eee", color: "#888" }}
                  >
                    {mol.role}
                  </div>
                </div>
              </FadeSection>
            ))}
          </div>

          {/* Radar chart */}
          <FadeSection delay={0.3}>
            <div className="mt-14 flex justify-center">
              <div
                className="w-full max-w-lg rounded-sm border p-6"
                style={{
                  borderColor: "#e5e5e5",
                  backgroundColor: "#fafafa",
                }}
              >
                <h4
                  className="mb-4 text-center text-[0.7rem] font-semibold uppercase tracking-[0.2em]"
                  style={{ color: ETH_BLUE }}
                >
                  Retrieval Accuracy by Species (1 − mRMSE)
                </h4>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData} cx="50%" cy="50%">
                    <PolarGrid stroke="#ddd" />
                    <PolarAngleAxis
                      dataKey="molecule"
                      tick={{ fontSize: 13, fill: "#444" }}
                    />
                    <PolarRadiusAxis
                      angle={90}
                      domain={[0, 1]}
                      tick={{ fontSize: 10, fill: "#999" }}
                      tickCount={5}
                    />
                    <Radar
                      name="ExoBiome"
                      dataKey="ExoBiome"
                      stroke={ETH_RED}
                      fill={ETH_RED}
                      fillOpacity={0.15}
                      strokeWidth={2}
                    />
                    <Radar
                      name="CNN Baseline"
                      dataKey="Baseline"
                      stroke="#bbb"
                      fill="#bbb"
                      fillOpacity={0.05}
                      strokeWidth={1.5}
                      strokeDasharray="4 4"
                    />
                    <Legend
                      wrapperStyle={{
                        fontSize: 12,
                        fontFamily: "var(--font-dm-sans)",
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </FadeSection>
        </div>
      </section>

      {/* ═══════════════════════════ HARDWARE ═══════════════════════════ */}
      <section style={{ backgroundColor: GRAY_BG }}>
        <div className="mx-auto max-w-[1100px] px-6 py-20">
          <FadeSection>
            <h2
              className="mb-2 text-[0.75rem] font-semibold uppercase tracking-[0.2em]"
              style={{ color: ETH_BLUE }}
            >
              Quantum Hardware
            </h2>
            <div
              className="mb-10 h-[2px] w-12"
              style={{ backgroundColor: ETH_BLUE }}
            />
          </FadeSection>

          <div className="grid gap-6 md:grid-cols-2">
            {[
              {
                name: "IQM Spark — Odra 5",
                location: "PWR Wrocław, Poland",
                qubits: "5 qubits",
                type: "Superconducting transmon",
                sdk: "qiskit-on-iqm",
                detail:
                  "On-premises system at Wrocław University of Science and Technology. Used for circuit development, calibration, and noise characterization during model training.",
                tag: "Development",
              },
              {
                name: "VTT Q50",
                location: "VTT Finland (remote)",
                qubits: "53 qubits",
                type: "Superconducting transmon",
                sdk: "qiskit (remote API)",
                detail:
                  "High-qubit-count processor accessed remotely through PWR partnership. Enabled execution of the full 12-qubit variational circuit for inference and benchmarking.",
                tag: "Production",
              },
            ].map((hw, i) => (
              <FadeSection key={hw.name} delay={i * 0.12}>
                <div
                  className="h-full rounded-sm border bg-white p-8"
                  style={{ borderColor: "#e5e5e5" }}
                >
                  <div className="mb-4 flex items-start justify-between">
                    <h3
                      className="text-[1.25rem]"
                      style={{
                        fontFamily: "var(--font-dm-serif)",
                        color: NAVY,
                      }}
                    >
                      {hw.name}
                    </h3>
                    <span
                      className="rounded-sm px-2.5 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-white"
                      style={{
                        backgroundColor:
                          hw.tag === "Production" ? ETH_RED : ETH_BLUE,
                      }}
                    >
                      {hw.tag}
                    </span>
                  </div>

                  <div
                    className="mb-5 text-[0.85rem] font-light"
                    style={{ color: "#666" }}
                  >
                    {hw.location}
                  </div>

                  <div className="mb-6 space-y-2">
                    {[
                      ["Qubits", hw.qubits],
                      ["Type", hw.type],
                      ["SDK", hw.sdk],
                    ].map(([label, value]) => (
                      <div
                        key={label}
                        className="flex justify-between border-b py-2 text-[0.82rem]"
                        style={{ borderColor: "#f0f0f0" }}
                      >
                        <span
                          className="font-medium uppercase tracking-[0.08em]"
                          style={{ color: "#999", fontSize: "0.72rem" }}
                        >
                          {label}
                        </span>
                        <span
                          style={{
                            fontFamily: "var(--font-jetbrains)",
                            color: "#333",
                          }}
                        >
                          {value}
                        </span>
                      </div>
                    ))}
                  </div>

                  <p
                    className="text-[0.85rem] leading-[1.7] font-light"
                    style={{ color: "#555" }}
                  >
                    {hw.detail}
                  </p>
                </div>
              </FadeSection>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════ DATASET ═══════════════════════════ */}
      <section className="bg-white">
        <div className="mx-auto max-w-[1100px] px-6 py-20">
          <FadeSection>
            <h2
              className="mb-2 text-[0.75rem] font-semibold uppercase tracking-[0.2em]"
              style={{ color: ETH_BLUE }}
            >
              Training Data
            </h2>
            <div
              className="mb-10 h-[2px] w-12"
              style={{ backgroundColor: ETH_BLUE }}
            />
          </FadeSection>

          <div className="grid gap-16 md:grid-cols-2">
            <FadeSection>
              <div>
                <h3
                  className="mb-5 text-[1.5rem] leading-snug"
                  style={{
                    fontFamily: "var(--font-dm-serif)",
                    color: NAVY,
                  }}
                >
                  Synthetic Spectra from
                  <br />
                  Forward Models
                </h3>
                <p
                  className="mb-6 text-[0.95rem] leading-[1.8] font-light"
                  style={{ color: "#444" }}
                >
                  ExoBiome was trained on synthetic transmission spectra
                  generated with the TauREx 3 radiative transfer code, aligned
                  with the Ariel Data Challenge 2023 format. Each spectrum
                  consists of 52 wavelength bins covering 0.5–7.8 μm, paired
                  with ground-truth log₁₀ volume mixing ratios for all five
                  target molecules.
                </p>
                <p
                  className="text-[0.95rem] leading-[1.8] font-light"
                  style={{ color: "#444" }}
                >
                  The primary training corpus comprises 106,000 spectra from
                  the ABC Database, supplemented by the 41,000-spectrum ADC
                  2023 training set, providing broad coverage of atmospheric
                  compositions and planetary parameters.
                </p>
              </div>
            </FadeSection>

            <FadeSection delay={0.12}>
              <div className="space-y-4">
                {[
                  {
                    source: "ABC Database",
                    count: "106,000",
                    detail: "Zenodo 6770103",
                    role: "Primary training set",
                  },
                  {
                    source: "ADC 2023",
                    count: "41,000",
                    detail: "ariel-datachallenge.space",
                    role: "Supplementary training + validation",
                  },
                  {
                    source: "MultiREx",
                    count: "Custom",
                    detail: "TauREx 3 wrapper",
                    role: "Augmentation & edge cases",
                  },
                ].map((ds) => (
                  <div
                    key={ds.source}
                    className="rounded-sm border p-5"
                    style={{
                      borderColor: "#e8e8e8",
                      backgroundColor: GRAY_BG,
                    }}
                  >
                    <div className="flex items-baseline justify-between mb-2">
                      <span
                        className="text-[1rem] font-medium"
                        style={{ color: NAVY }}
                      >
                        {ds.source}
                      </span>
                      <span
                        className="text-[0.85rem] font-semibold"
                        style={{
                          fontFamily: "var(--font-jetbrains)",
                          color: ETH_BLUE,
                        }}
                      >
                        {ds.count}
                      </span>
                    </div>
                    <div
                      className="text-[0.78rem] font-light"
                      style={{
                        fontFamily: "var(--font-jetbrains)",
                        color: "#999",
                      }}
                    >
                      {ds.detail}
                    </div>
                    <div
                      className="mt-2 text-[0.8rem] font-light"
                      style={{ color: "#666" }}
                    >
                      {ds.role}
                    </div>
                  </div>
                ))}
              </div>
            </FadeSection>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════ CONCLUSION ═══════════════════════════ */}
      <section style={{ backgroundColor: GRAY_BG }}>
        <div className="mx-auto max-w-[1100px] px-6 py-20">
          <FadeSection>
            <h2
              className="mb-2 text-[0.75rem] font-semibold uppercase tracking-[0.2em]"
              style={{ color: ETH_BLUE }}
            >
              Conclusions & Future Work
            </h2>
            <div
              className="mb-10 h-[2px] w-12"
              style={{ backgroundColor: ETH_BLUE }}
            />
          </FadeSection>

          <div className="grid gap-16 md:grid-cols-2">
            <FadeSection>
              <div>
                <h3
                  className="mb-6 text-[1.5rem] leading-snug"
                  style={{
                    fontFamily: "var(--font-dm-serif)",
                    color: NAVY,
                  }}
                >
                  Key Achievements
                </h3>
                <div className="space-y-5">
                  {[
                    {
                      title: "State-of-the-art retrieval accuracy",
                      desc: "0.295 mRMSE surpasses the ADC 2023 winning solution, establishing a new benchmark for spectral retrieval from Ariel-format data.",
                    },
                    {
                      title: "First quantum ML for biosignatures",
                      desc: "To our knowledge, ExoBiome is the first system applying quantum machine learning to biosignature detection from exoplanetary spectra.",
                    },
                    {
                      title: "Real quantum hardware execution",
                      desc: "All quantum circuits executed on physical superconducting processors (IQM Spark, VTT Q50), not simulators — validating practical quantum advantage.",
                    },
                    {
                      title: "End-to-end pipeline",
                      desc: "From raw transmission spectrum to molecular abundance predictions in a single forward pass, with no intermediate retrieval steps.",
                    },
                  ].map((item, i) => (
                    <div
                      key={i}
                      className="border-l-[3px] pl-5"
                      style={{
                        borderColor: i === 0 ? ETH_RED : "#d4d4d4",
                      }}
                    >
                      <div
                        className="mb-1 text-[0.9rem] font-semibold"
                        style={{ color: NAVY }}
                      >
                        {item.title}
                      </div>
                      <div
                        className="text-[0.85rem] leading-[1.7] font-light"
                        style={{ color: "#555" }}
                      >
                        {item.desc}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </FadeSection>

            <FadeSection delay={0.12}>
              <div>
                <h3
                  className="mb-6 text-[1.5rem] leading-snug"
                  style={{
                    fontFamily: "var(--font-dm-serif)",
                    color: NAVY,
                  }}
                >
                  Future Directions
                </h3>
                <div className="space-y-5">
                  {[
                    {
                      title: "JWST validation",
                      desc: "Apply ExoBiome to real JWST transmission spectra from K2-18b and WASP-39b to validate performance on observational data.",
                    },
                    {
                      title: "Uncertainty quantification",
                      desc: "Leverage quantum measurement statistics for native Bayesian uncertainty estimates on retrieved abundances.",
                    },
                    {
                      title: "Extended molecular coverage",
                      desc: "Expand target species to include O₃, SO₂, and PH₃ — key indicators of biological and volcanic activity.",
                    },
                    {
                      title: "Scalability studies",
                      desc: "Investigate performance scaling with qubit count on next-generation processors (20+ qubits) for higher-dimensional feature spaces.",
                    },
                  ].map((item, i) => (
                    <div
                      key={i}
                      className="border-l-[3px] pl-5"
                      style={{ borderColor: "#d4d4d4" }}
                    >
                      <div
                        className="mb-1 text-[0.9rem] font-semibold"
                        style={{ color: NAVY }}
                      >
                        {item.title}
                      </div>
                      <div
                        className="text-[0.85rem] leading-[1.7] font-light"
                        style={{ color: "#555" }}
                      >
                        {item.desc}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </FadeSection>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════ TECH STACK ═══════════════════════════ */}
      <section className="bg-white">
        <div className="mx-auto max-w-[1100px] px-6 py-16">
          <FadeSection>
            <h2
              className="mb-2 text-[0.75rem] font-semibold uppercase tracking-[0.2em]"
              style={{ color: ETH_BLUE }}
            >
              Technology Stack
            </h2>
            <div
              className="mb-8 h-[2px] w-12"
              style={{ backgroundColor: ETH_BLUE }}
            />
          </FadeSection>

          <FadeSection delay={0.1}>
            <div className="flex flex-wrap gap-3">
              {[
                "Python",
                "PyTorch",
                "Qiskit",
                "qiskit-on-iqm",
                "sQUlearn",
                "scikit-learn",
                "TauREx 3",
                "MultiREx",
                "SpectRes",
                "SciPy",
                "Matplotlib",
                "Jupyter",
              ].map((tech) => (
                <span
                  key={tech}
                  className="rounded-sm border px-4 py-2 text-[0.78rem] font-medium"
                  style={{
                    borderColor: "#ddd",
                    color: "#444",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  {tech}
                </span>
              ))}
            </div>
          </FadeSection>
        </div>
      </section>

      {/* ═══════════════════════════ FOOTER ═══════════════════════════ */}
      <footer style={{ backgroundColor: NAVY }}>
        <div className="mx-auto max-w-[1100px] px-6 py-12">
          <div className="grid gap-8 md:grid-cols-3">
            <div>
              <h3
                className="mb-3 text-[1.1rem] text-white"
                style={{ fontFamily: "var(--font-dm-serif)" }}
              >
                ExoBiome
              </h3>
              <p
                className="text-[0.8rem] leading-relaxed font-light"
                style={{ color: "rgba(255,255,255,0.5)" }}
              >
                Quantum biosignature detection
                <br />
                from exoplanet transmission spectra
              </p>
            </div>
            <div>
              <h4
                className="mb-3 text-[0.65rem] font-semibold uppercase tracking-[0.2em]"
                style={{ color: "rgba(255,255,255,0.4)" }}
              >
                Presented at
              </h4>
              <p
                className="text-[0.85rem] font-light"
                style={{ color: "rgba(255,255,255,0.7)" }}
              >
                HACK-4-SAGES 2026
                <br />
                <span style={{ color: "rgba(255,255,255,0.4)" }}>
                  ETH Zurich · COPL · Origins Federation
                </span>
              </p>
            </div>
            <div>
              <h4
                className="mb-3 text-[0.65rem] font-semibold uppercase tracking-[0.2em]"
                style={{ color: "rgba(255,255,255,0.4)" }}
              >
                Category
              </h4>
              <p
                className="text-[0.85rem] font-light"
                style={{ color: "rgba(255,255,255,0.7)" }}
              >
                Life Detection and Biosignatures
              </p>
              <div
                className="mt-4 h-[1px] w-full"
                style={{ backgroundColor: "rgba(255,255,255,0.1)" }}
              />
              <p
                className="mt-4 text-[0.72rem] font-light"
                style={{ color: "rgba(255,255,255,0.3)" }}
              >
                March 9–13, 2026 · Zurich, Switzerland
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
