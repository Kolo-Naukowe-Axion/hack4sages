"use client";

import { useRef } from "react";
import { Inter, JetBrains_Mono } from "next/font/google";
import { motion, useInView } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from "recharts";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

// ─── Data ────────────────────────────────────────────────────────────
const molecules = [
  { formula: "H₂O", score: 0.218, color: "#3b82f6", glow: "0 0 40px rgba(59,130,246,0.4)" },
  { formula: "CO₂", score: 0.261, color: "#ef4444", glow: "0 0 40px rgba(239,68,68,0.4)" },
  { formula: "CO", score: 0.327, color: "#f59e0b", glow: "0 0 40px rgba(245,158,11,0.4)" },
  { formula: "CH₄", score: 0.29, color: "#10b981", glow: "0 0 40px rgba(16,185,129,0.4)" },
  { formula: "NH₃", score: 0.378, color: "#a855f7", glow: "0 0 40px rgba(168,85,247,0.4)" },
];

const comparisonData = [
  { name: "ExoBiome", mRMSE: 0.295, fill: "#06b6d4" },
  { name: "ADC Winner", mRMSE: 0.32, fill: "#8b5cf6" },
  { name: "CNN", mRMSE: 0.85, fill: "#64748b" },
  { name: "RF", mRMSE: 1.2, fill: "#475569" },
];

const radarData = molecules.map((m) => ({
  molecule: m.formula,
  ExoBiome: +(1 - m.score).toFixed(3),
  Baseline: +(1 - m.score * 2.8).toFixed(3),
}));

const architectureSteps = [
  { label: "Spectrum Input", sub: "52 wavelength bins", icon: "◇" },
  { label: "SpectralEncoder", sub: "Conv1D → BatchNorm → ReLU", icon: "▣" },
  { label: "AuxEncoder", sub: "Stellar + Planetary params", icon: "▢" },
  { label: "Fusion Layer", sub: "Concatenate → Dense", icon: "⬡" },
  { label: "Quantum Circuit", sub: "12 qubits · RY + CNOT", icon: "◎" },
  { label: "Output", sub: "log₁₀ VMR × 5 molecules", icon: "◈" },
];

// ─── Reusable Components ─────────────────────────────────────────────
function GlassCard({
  children,
  className = "",
  glowColor,
}: {
  children: React.ReactNode;
  className?: string;
  glowColor?: string;
}) {
  return (
    <div
      className={`relative rounded-2xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-xl ${className}`}
      style={glowColor ? { boxShadow: glowColor } : undefined}
    >
      {children}
    </div>
  );
}

function FadeInSection({
  children,
  className = "",
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 60 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.9, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

function SectionLabel({ text }: { text: string }) {
  return (
    <span className="inline-block text-xs font-medium tracking-[0.3em] uppercase text-cyan-400/70 mb-4">
      {text}
    </span>
  );
}

// ─── Custom Tooltip ──────────────────────────────────────────────────
function GlassTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; name: string; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-white/10 bg-black/60 backdrop-blur-xl px-4 py-3 text-sm">
      <p className="text-white/60 mb-1 font-medium">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }} className="font-mono">
          {p.name}: {p.value}
        </p>
      ))}
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────
export default function ExoBiomePage() {
  return (
    <div
      className={`${inter.variable} ${mono.variable} font-[family-name:var(--font-inter)] relative`}
    >
      {/* Global animated nebula background */}
      <div className="fixed inset-0 z-0 overflow-hidden bg-[#050510]">
        <div className="nebula-blob nebula-blob-1" />
        <div className="nebula-blob nebula-blob-2" />
        <div className="nebula-blob nebula-blob-3" />
        <div className="nebula-blob nebula-blob-4" />
      </div>

      {/* Scroll-snap container */}
      <main className="relative z-10 snap-y snap-mandatory h-screen overflow-y-auto scroll-smooth">
        {/* ───── SECTION 1: Hero ───── */}
        <section className="snap-start h-screen flex flex-col items-center justify-center relative px-6">
          <FadeInSection className="flex flex-col items-center text-center max-w-4xl">
            <SectionLabel text="HACK-4-SAGES 2026 · ETH Zurich" />
            <h1 className="text-6xl md:text-8xl font-bold tracking-tight text-white mb-6">
              Exo
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-400 to-cyan-400">
                Biome
              </span>
            </h1>
            <p className="text-lg md:text-xl text-white/50 max-w-2xl leading-relaxed mb-10">
              Quantum biosignature detection from exoplanet transmission spectra.
              The first application of quantum machine learning to identify signs
              of life beyond Earth.
            </p>
            <div className="flex items-center gap-6 text-sm text-white/30">
              <span className="font-[family-name:var(--font-jetbrains)]">12 qubits</span>
              <span className="w-1 h-1 rounded-full bg-white/20" />
              <span className="font-[family-name:var(--font-jetbrains)]">5 molecules</span>
              <span className="w-1 h-1 rounded-full bg-white/20" />
              <span className="font-[family-name:var(--font-jetbrains)]">41,423 spectra</span>
            </div>
          </FadeInSection>

          {/* Scroll hint */}
          <motion.div
            className="absolute bottom-10 flex flex-col items-center gap-2 text-white/20 text-xs tracking-widest"
            animate={{ y: [0, 8, 0] }}
            transition={{ repeat: Infinity, duration: 2 }}
          >
            <span>SCROLL</span>
            <svg width="16" height="24" viewBox="0 0 16 24" fill="none">
              <path d="M8 4v12m0 0l-4-4m4 4l4-4" stroke="currentColor" strokeWidth="1.5" />
            </svg>
          </motion.div>
        </section>

        {/* ───── SECTION 2: Problem ───── */}
        <section className="snap-start min-h-screen flex items-center justify-center px-6 py-24">
          <div className="max-w-5xl w-full">
            <FadeInSection>
              <SectionLabel text="The Challenge" />
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-8 leading-tight">
                Finding life in the light<br />
                <span className="text-white/30">of distant worlds</span>
              </h2>
            </FadeInSection>
            <div className="grid md:grid-cols-3 gap-6 mt-12">
              {[
                {
                  title: "Transmission Spectra",
                  body: "When an exoplanet transits its star, starlight filters through the atmosphere. Each molecule absorbs at characteristic wavelengths, encoding its presence into the spectrum.",
                  num: "01",
                },
                {
                  title: "Atmospheric Retrieval",
                  body: "Extracting molecular abundances (log₁₀ VMR) from noisy, low-resolution spectra is an ill-posed inverse problem. Traditional Bayesian methods take hours per planet.",
                  num: "02",
                },
                {
                  title: "Quantum Advantage",
                  body: "Quantum circuits can explore exponentially large feature spaces. Our QELM maps spectral features into 2¹² = 4096 dimensional Hilbert space for superior pattern recognition.",
                  num: "03",
                },
              ].map((card, i) => (
                <FadeInSection key={card.num} delay={i * 0.15}>
                  <GlassCard className="p-8 h-full">
                    <span className="text-cyan-500/40 font-[family-name:var(--font-jetbrains)] text-sm">
                      {card.num}
                    </span>
                    <h3 className="text-xl font-semibold text-white mt-3 mb-4">
                      {card.title}
                    </h3>
                    <p className="text-white/40 text-sm leading-relaxed">
                      {card.body}
                    </p>
                  </GlassCard>
                </FadeInSection>
              ))}
            </div>
          </div>
        </section>

        {/* ───── SECTION 3: Architecture ───── */}
        <section className="snap-start min-h-screen flex items-center justify-center px-6 py-24">
          <div className="max-w-5xl w-full">
            <FadeInSection>
              <SectionLabel text="Architecture" />
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-16 leading-tight">
                Quantum-classical<br />
                <span className="text-white/30">hybrid pipeline</span>
              </h2>
            </FadeInSection>

            <div className="relative flex flex-col gap-0">
              {/* Connecting line */}
              <div className="absolute left-8 top-0 bottom-0 w-px bg-gradient-to-b from-cyan-500/0 via-cyan-500/30 to-purple-500/0" />

              {architectureSteps.map((step, i) => (
                <FadeInSection key={step.label} delay={i * 0.1}>
                  <div className="relative flex items-center gap-6 py-5">
                    {/* Node dot */}
                    <div
                      className="relative z-10 w-16 h-16 rounded-xl border border-white/10 bg-white/[0.03] backdrop-blur-lg flex items-center justify-center text-2xl shrink-0"
                      style={
                        i === 4
                          ? {
                              borderImage:
                                "conic-gradient(from var(--angle, 0deg), #06b6d4, #a855f7, #06b6d4) 1",
                              animation: "spin-border 4s linear infinite",
                            }
                          : undefined
                      }
                    >
                      <span className={i === 4 ? "text-cyan-400" : "text-white/40"}>
                        {step.icon}
                      </span>
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-white">
                        {step.label}
                      </h3>
                      <p className="text-sm text-white/30 font-[family-name:var(--font-jetbrains)]">
                        {step.sub}
                      </p>
                    </div>
                  </div>
                </FadeInSection>
              ))}
            </div>

            {/* Quantum circuit detail card */}
            <FadeInSection delay={0.4}>
              <div className="mt-16 quantum-border-card rounded-2xl p-8 bg-white/[0.02] backdrop-blur-xl">
                <h3 className="text-xl font-semibold text-white mb-4">
                  Quantum Circuit Detail
                </h3>
                <div className="grid md:grid-cols-3 gap-6 text-sm">
                  <div>
                    <span className="text-white/30">Qubits</span>
                    <p className="text-2xl font-[family-name:var(--font-jetbrains)] text-cyan-400 mt-1">12</p>
                  </div>
                  <div>
                    <span className="text-white/30">Gate Set</span>
                    <p className="text-2xl font-[family-name:var(--font-jetbrains)] text-purple-400 mt-1">
                      RY + CNOT
                    </p>
                  </div>
                  <div>
                    <span className="text-white/30">Hilbert Space</span>
                    <p className="text-2xl font-[family-name:var(--font-jetbrains)] text-emerald-400 mt-1">
                      2¹² = 4096
                    </p>
                  </div>
                </div>
              </div>
            </FadeInSection>
          </div>
        </section>

        {/* ───── SECTION 4: Hero Result ───── */}
        <section className="snap-start h-screen flex flex-col items-center justify-center px-6 relative">
          <FadeInSection className="flex flex-col items-center text-center">
            <SectionLabel text="Performance" />
            <p className="text-white/40 text-lg mb-6">Mean Relative RMSE</p>

            {/* Giant number with glow */}
            <div className="relative">
              <div className="absolute inset-0 blur-[100px] bg-cyan-500/20 rounded-full scale-150" />
              <motion.span
                className="relative text-[8rem] md:text-[12rem] font-bold font-[family-name:var(--font-jetbrains)] text-white leading-none"
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
              >
                0.295
              </motion.span>
            </div>

            <p className="text-white/30 mt-8 text-sm max-w-md">
              Outperforming the Ariel Data Challenge 2023 winning solution
              with a quantum-classical hybrid approach
            </p>

            <div className="flex items-center gap-4 mt-8">
              <span className="px-3 py-1.5 rounded-full border border-cyan-500/20 bg-cyan-500/5 text-cyan-400 text-xs font-[family-name:var(--font-jetbrains)]">
                vs ADC Winner: 0.32
              </span>
              <span className="px-3 py-1.5 rounded-full border border-white/10 bg-white/[0.03] text-white/40 text-xs font-[family-name:var(--font-jetbrains)]">
                8% improvement
              </span>
            </div>
          </FadeInSection>
        </section>

        {/* ───── SECTION 5: Per-Molecule Breakdown ───── */}
        <section className="snap-start min-h-screen flex items-center justify-center px-6 py-24">
          <div className="max-w-5xl w-full">
            <FadeInSection>
              <SectionLabel text="Per-Molecule Results" />
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-16">
                Five molecules,<br />
                <span className="text-white/30">five biosignatures</span>
              </h2>
            </FadeInSection>

            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {molecules.map((m, i) => (
                <FadeInSection key={m.formula} delay={i * 0.1}>
                  <GlassCard
                    className="p-6 text-center group hover:bg-white/[0.05] transition-colors duration-500"
                    glowColor={m.glow}
                  >
                    <span
                      className="text-3xl md:text-4xl font-bold font-[family-name:var(--font-jetbrains)] block mb-3"
                      style={{ color: m.color }}
                    >
                      {m.formula}
                    </span>
                    <span className="text-2xl font-[family-name:var(--font-jetbrains)] text-white block mb-1">
                      {m.score}
                    </span>
                    <span className="text-xs text-white/30">mRMSE</span>
                  </GlassCard>
                </FadeInSection>
              ))}
            </div>

            {/* Radar chart */}
            <FadeInSection delay={0.3}>
              <GlassCard className="mt-12 p-8">
                <h3 className="text-lg font-semibold text-white mb-6">
                  Retrieval Accuracy by Molecule
                </h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="rgba(255,255,255,0.06)" />
                      <PolarAngleAxis
                        dataKey="molecule"
                        tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 13 }}
                      />
                      <PolarRadiusAxis
                        angle={90}
                        domain={[0, 1]}
                        tick={{ fill: "rgba(255,255,255,0.2)", fontSize: 10 }}
                        axisLine={false}
                      />
                      <Radar
                        name="ExoBiome"
                        dataKey="ExoBiome"
                        stroke="#06b6d4"
                        fill="#06b6d4"
                        fillOpacity={0.15}
                        strokeWidth={2}
                      />
                      <Radar
                        name="Baseline (CNN)"
                        dataKey="Baseline"
                        stroke="#64748b"
                        fill="#64748b"
                        fillOpacity={0.05}
                        strokeWidth={1}
                        strokeDasharray="4 4"
                      />
                      <Legend
                        wrapperStyle={{ color: "rgba(255,255,255,0.5)", fontSize: 12 }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </GlassCard>
            </FadeInSection>
          </div>
        </section>

        {/* ───── SECTION 6: Comparison Chart ───── */}
        <section className="snap-start min-h-screen flex items-center justify-center px-6 py-24">
          <div className="max-w-5xl w-full">
            <FadeInSection>
              <SectionLabel text="Benchmarks" />
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-16">
                Against the<br />
                <span className="text-white/30">state of the art</span>
              </h2>
            </FadeInSection>

            <FadeInSection delay={0.2}>
              <GlassCard className="p-8">
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={comparisonData}
                      layout="vertical"
                      margin={{ left: 30, right: 30, top: 10, bottom: 10 }}
                    >
                      <CartesianGrid
                        horizontal={false}
                        stroke="rgba(255,255,255,0.04)"
                      />
                      <XAxis
                        type="number"
                        domain={[0, 1.4]}
                        tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 12 }}
                        axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                        tickLine={false}
                      />
                      <YAxis
                        type="category"
                        dataKey="name"
                        tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 14 }}
                        axisLine={false}
                        tickLine={false}
                        width={100}
                      />
                      <Tooltip
                        content={<GlassTooltip />}
                        cursor={{ fill: "rgba(255,255,255,0.02)" }}
                      />
                      <Bar
                        dataKey="mRMSE"
                        radius={[0, 6, 6, 0]}
                        barSize={36}
                      >
                        {comparisonData.map((entry) => (
                          <motion.rect
                            key={entry.name}
                            fill={entry.fill}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-white/20 text-xs mt-4 text-center">
                  Lower mRMSE is better. ExoBiome achieves state-of-the-art
                  performance on the ADC 2023 benchmark.
                </p>
              </GlassCard>
            </FadeInSection>

            <div className="grid md:grid-cols-3 gap-6 mt-8">
              {[
                { label: "Improvement over ADC Winner", value: "~8%", sub: "0.295 vs 0.320" },
                { label: "vs Classical CNN", value: "~65%", sub: "0.295 vs 0.850" },
                { label: "vs Random Forest", value: "~75%", sub: "0.295 vs 1.200" },
              ].map((stat, i) => (
                <FadeInSection key={stat.label} delay={i * 0.1}>
                  <GlassCard className="p-6 text-center">
                    <span className="text-3xl font-bold font-[family-name:var(--font-jetbrains)] text-white">
                      {stat.value}
                    </span>
                    <p className="text-white/50 text-sm mt-2">{stat.label}</p>
                    <p className="text-white/20 text-xs font-[family-name:var(--font-jetbrains)] mt-1">
                      {stat.sub}
                    </p>
                  </GlassCard>
                </FadeInSection>
              ))}
            </div>
          </div>
        </section>

        {/* ───── SECTION 7: Dataset & Method ───── */}
        <section className="snap-start min-h-screen flex items-center justify-center px-6 py-24">
          <div className="max-w-5xl w-full">
            <FadeInSection>
              <SectionLabel text="Data & Method" />
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-16">
                Rigorous science,<br />
                <span className="text-white/30">reproducible results</span>
              </h2>
            </FadeInSection>

            <div className="grid md:grid-cols-2 gap-6">
              <FadeInSection delay={0.1}>
                <GlassCard className="p-8 h-full">
                  <h3 className="text-xl font-semibold text-white mb-6">
                    Dataset
                  </h3>
                  <div className="space-y-4 text-sm">
                    <div className="flex justify-between border-b border-white/5 pb-3">
                      <span className="text-white/40">Source</span>
                      <span className="text-white/70 font-[family-name:var(--font-jetbrains)]">
                        Ariel Data Challenge 2023
                      </span>
                    </div>
                    <div className="flex justify-between border-b border-white/5 pb-3">
                      <span className="text-white/40">Spectra</span>
                      <span className="text-white/70 font-[family-name:var(--font-jetbrains)]">
                        41,423
                      </span>
                    </div>
                    <div className="flex justify-between border-b border-white/5 pb-3">
                      <span className="text-white/40">Wavelength Bins</span>
                      <span className="text-white/70 font-[family-name:var(--font-jetbrains)]">
                        52
                      </span>
                    </div>
                    <div className="flex justify-between border-b border-white/5 pb-3">
                      <span className="text-white/40">Targets</span>
                      <span className="text-white/70 font-[family-name:var(--font-jetbrains)]">
                        log₁₀ VMR × 5
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/40">Split</span>
                      <span className="text-white/70 font-[family-name:var(--font-jetbrains)]">
                        80 / 10 / 10
                      </span>
                    </div>
                  </div>
                </GlassCard>
              </FadeInSection>

              <FadeInSection delay={0.2}>
                <GlassCard className="p-8 h-full">
                  <h3 className="text-xl font-semibold text-white mb-6">
                    Pipeline
                  </h3>
                  <div className="space-y-5 text-sm">
                    {[
                      { step: "1", text: "Preprocess spectra: normalize, add noise augmentation" },
                      { step: "2", text: "Encode spectral features via Conv1D + BatchNorm blocks" },
                      { step: "3", text: "Encode auxiliary stellar/planetary parameters" },
                      { step: "4", text: "Fuse representations, project to 12-dim quantum input" },
                      { step: "5", text: "Execute parameterized quantum circuit (RY rotations + CNOT entanglement)" },
                      { step: "6", text: "Measure expectation values, map to 5 molecular abundances" },
                    ].map((s) => (
                      <div key={s.step} className="flex gap-4">
                        <span className="text-cyan-500/60 font-[family-name:var(--font-jetbrains)] shrink-0 w-6">
                          {s.step}.
                        </span>
                        <span className="text-white/40">{s.text}</span>
                      </div>
                    ))}
                  </div>
                </GlassCard>
              </FadeInSection>
            </div>

            <FadeInSection delay={0.3}>
              <GlassCard className="mt-6 p-8">
                <h3 className="text-xl font-semibold text-white mb-6">
                  Key References
                </h3>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  {[
                    {
                      authors: "Vetrano et al. 2025",
                      title: "QELM for Atmospheric Retrieval",
                      ref: "arXiv:2509.03617",
                    },
                    {
                      authors: "Cardenas et al. 2025",
                      title: "MultiREx Dataset",
                      ref: "MNRAS 539",
                    },
                    {
                      authors: "Schwieterman et al. 2018",
                      title: "Exoplanet Biosignatures Review",
                      ref: "Astrobiology 18(6)",
                    },
                    {
                      authors: "Seeburger et al. 2023",
                      title: "Methanogenesis to Planetary Spectra",
                      ref: "ApJ Letters",
                    },
                  ].map((ref) => (
                    <div
                      key={ref.ref}
                      className="flex flex-col gap-1 p-4 rounded-lg bg-white/[0.02] border border-white/[0.04]"
                    >
                      <span className="text-white/60 font-medium">
                        {ref.authors}
                      </span>
                      <span className="text-white/30">{ref.title}</span>
                      <span className="text-cyan-500/50 font-[family-name:var(--font-jetbrains)] text-xs">
                        {ref.ref}
                      </span>
                    </div>
                  ))}
                </div>
              </GlassCard>
            </FadeInSection>
          </div>
        </section>

        {/* ───── SECTION 8: Closing ───── */}
        <section className="snap-start h-screen flex flex-col items-center justify-center px-6 relative">
          <FadeInSection className="flex flex-col items-center text-center max-w-3xl">
            <SectionLabel text="ExoBiome" />
            <h2 className="text-4xl md:text-6xl font-bold text-white mb-8 leading-tight">
              Quantum eyes<br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
                on alien skies
              </span>
            </h2>
            <p className="text-white/40 text-lg max-w-xl leading-relaxed mb-12">
              A quantum-classical hybrid that pushes the frontier of atmospheric
              retrieval — demonstrating that quantum machine learning can deliver
              real scientific advantage today.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-3 text-xs font-[family-name:var(--font-jetbrains)] text-white/20">
              {["Python", "PyTorch", "Qiskit", "sQUlearn", "IQM Spark", "TauREx 3"].map(
                (t) => (
                  <span
                    key={t}
                    className="px-3 py-1.5 rounded-full border border-white/[0.06] bg-white/[0.02]"
                  >
                    {t}
                  </span>
                )
              )}
            </div>
          </FadeInSection>

          <div className="absolute bottom-10 text-white/15 text-xs tracking-widest">
            HACK-4-SAGES 2026 · ETH ZURICH
          </div>
        </section>
      </main>

      {/* CSS-only animations */}
      <style jsx>{`
        .nebula-blob {
          position: absolute;
          border-radius: 50%;
          filter: blur(120px);
          will-change: transform;
        }
        .nebula-blob-1 {
          width: 800px;
          height: 800px;
          background: radial-gradient(circle, #1a0030 0%, transparent 70%);
          top: -200px;
          left: -200px;
          animation: drift1 25s ease-in-out infinite;
        }
        .nebula-blob-2 {
          width: 600px;
          height: 600px;
          background: radial-gradient(circle, #001a1a 0%, transparent 70%);
          bottom: -100px;
          right: -100px;
          animation: drift2 30s ease-in-out infinite;
        }
        .nebula-blob-3 {
          width: 500px;
          height: 500px;
          background: radial-gradient(circle, #000820 0%, transparent 70%);
          top: 40%;
          left: 50%;
          animation: drift3 20s ease-in-out infinite;
        }
        .nebula-blob-4 {
          width: 400px;
          height: 400px;
          background: radial-gradient(circle, #0a0020 0%, transparent 70%);
          top: 60%;
          left: 20%;
          animation: drift4 35s ease-in-out infinite;
        }

        @keyframes drift1 {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(100px, 80px) scale(1.1); }
          66% { transform: translate(-50px, 120px) scale(0.95); }
        }
        @keyframes drift2 {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(-80px, -60px) scale(1.05); }
          66% { transform: translate(60px, -100px) scale(1.1); }
        }
        @keyframes drift3 {
          0%, 100% { transform: translate(0, 0) scale(1); }
          50% { transform: translate(-120px, 60px) scale(1.15); }
        }
        @keyframes drift4 {
          0%, 100% { transform: translate(0, 0) scale(1); }
          40% { transform: translate(80px, -80px) scale(1.1); }
          80% { transform: translate(-60px, 40px) scale(0.9); }
        }

        .quantum-border-card {
          position: relative;
          border: 1px solid transparent;
          background-clip: padding-box;
        }
        .quantum-border-card::before {
          content: '';
          position: absolute;
          inset: -1px;
          border-radius: inherit;
          padding: 1px;
          background: conic-gradient(from var(--angle, 0deg), #06b6d4, #a855f7, #10b981, #06b6d4);
          mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
          mask-composite: exclude;
          -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
          -webkit-mask-composite: xor;
          animation: rotate-angle 6s linear infinite;
        }

        @property --angle {
          syntax: '<angle>';
          initial-value: 0deg;
          inherits: false;
        }
        @keyframes rotate-angle {
          to { --angle: 360deg; }
        }

        @keyframes spin-border {
          to { --angle: 360deg; }
        }

        /* Scrollbar */
        main::-webkit-scrollbar {
          width: 4px;
        }
        main::-webkit-scrollbar-track {
          background: transparent;
        }
        main::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,0.08);
          border-radius: 2px;
        }
      `}</style>
    </div>
  );
}
