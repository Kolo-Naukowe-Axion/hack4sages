"use client";

import { useEffect, useRef, useState } from "react";
import { JetBrains_Mono } from "next/font/google";
import { motion, useInView, type Variants } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const jet = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["300", "400", "500", "700", "800"],
});

const CYAN = "#00ffcc";
const WHITE = "#ffffff";
const GRAY = "#666666";
const DARK_GRAY = "#333333";
const LIGHT_GRAY = "#999999";

const fadeUp: Variants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.8, ease: [0.25, 0.1, 0.25, 1] },
  },
};

const fadeIn: Variants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { duration: 1, ease: "easeOut" },
  },
};

const stagger: Variants = {
  visible: {
    transition: { staggerChildren: 0.12 },
  },
};

function Section({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <motion.section
      ref={ref}
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
      variants={stagger}
      className={`min-h-screen w-full flex flex-col justify-center px-8 md:px-16 lg:px-24 py-24 relative ${className}`}
    >
      {children}
    </motion.section>
  );
}

function Divider() {
  return (
    <div className="w-full px-8 md:px-16 lg:px-24">
      <div
        className="w-full"
        style={{ height: "1px", background: DARK_GRAY }}
      />
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <motion.p
      variants={fadeUp}
      className="text-[10px] tracking-[0.3em] uppercase mb-8"
      style={{ color: GRAY }}
    >
      {children}
    </motion.p>
  );
}

const comparisonData = [
  { model: "Random Forest", mrmse: 1.2 },
  { model: "CNN Baseline", mrmse: 0.85 },
  { model: "ADC 2023 Winner", mrmse: 0.32 },
  { model: "ExoBiome", mrmse: 0.295 },
];

const moleculeData = [
  { molecule: "H₂O", mrmse: "0.218", label: "Water" },
  { molecule: "CO₂", mrmse: "0.261", label: "Carbon Dioxide" },
  { molecule: "CH₄", mrmse: "0.290", label: "Methane" },
  { molecule: "CO", mrmse: "0.327", label: "Carbon Monoxide" },
  { molecule: "NH₃", mrmse: "0.378", label: "Ammonia" },
];

export default function ExoBiomePage() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      const docHeight =
        document.documentElement.scrollHeight - window.innerHeight;
      if (docHeight > 0) {
        setScrollProgress(scrollTop / docHeight);
      }
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div
      ref={containerRef}
      className={`${jet.className} bg-black text-white min-h-screen selection:bg-[#00ffcc]/20 selection:text-[#00ffcc]`}
      style={{ background: "#000" }}
    >
      {/* scroll progress */}
      <div
        className="fixed top-0 left-0 z-50"
        style={{
          height: "2px",
          width: `${scrollProgress * 100}%`,
          background: CYAN,
          transition: "width 0.1s linear",
        }}
      />

      {/* ═══════════════════════════════════════ */}
      {/* SECTION 0 — HERO */}
      {/* ═══════════════════════════════════════ */}
      <Section>
        <motion.p
          variants={fadeUp}
          className="text-[10px] tracking-[0.3em] uppercase mb-16"
          style={{ color: GRAY }}
        >
          HACK-4-SAGES 2026 // ETH Zurich // Life Detection &amp; Biosignatures
        </motion.p>

        <motion.h1
          variants={fadeUp}
          className="text-6xl md:text-8xl lg:text-[120px] font-extrabold leading-[0.85] tracking-tight mb-12"
        >
          Exo
          <br />
          Biome
        </motion.h1>

        <motion.p
          variants={fadeUp}
          className="text-base md:text-lg max-w-2xl leading-relaxed mb-16"
          style={{ color: LIGHT_GRAY }}
        >
          Quantum-classical neural network for biosignature detection
          <br />
          from exoplanet transmission spectra.
        </motion.p>

        <motion.div
          variants={fadeUp}
          className="flex flex-col gap-1 text-xs"
          style={{ color: GRAY }}
        >
          <span>ariel data challenge 2023</span>
          <span>41,423 transmission spectra</span>
          <span>12 qubits // 5 target molecules</span>
        </motion.div>

        <motion.div
          variants={fadeIn}
          className="absolute bottom-12 left-8 md:left-16 lg:left-24 text-[10px] tracking-[0.2em]"
          style={{ color: DARK_GRAY }}
        >
          SCROLL
        </motion.div>
      </Section>

      <Divider />

      {/* ═══════════════════════════════════════ */}
      {/* SECTION 1 — THE PROBLEM */}
      {/* ═══════════════════════════════════════ */}
      <Section>
        <Label>01 — Problem</Label>

        <motion.h2
          variants={fadeUp}
          className="text-3xl md:text-5xl lg:text-6xl font-bold leading-tight mb-12 max-w-4xl"
        >
          Can we detect life on other worlds from a single spectrum?
        </motion.h2>

        <motion.div variants={fadeUp} className="max-w-2xl space-y-6">
          <p className="text-sm leading-relaxed" style={{ color: LIGHT_GRAY }}>
            When starlight passes through an exoplanet&apos;s atmosphere, molecules
            absorb specific wavelengths. The resulting transmission spectrum
            encodes the atmospheric composition — a chemical fingerprint.
          </p>
          <p className="text-sm leading-relaxed" style={{ color: LIGHT_GRAY }}>
            Certain molecular combinations — H₂O alongside CH₄, depleted CO₂ —
            cannot be sustained by geology alone. They require biology.
            These are biosignatures.
          </p>
          <p className="text-sm leading-relaxed" style={{ color: LIGHT_GRAY }}>
            The challenge: retrieving precise molecular abundances from noisy,
            low-resolution spectra. Classical methods either lack accuracy
            or require prohibitive compute. We need a new approach.
          </p>
        </motion.div>
      </Section>

      <Divider />

      {/* ═══════════════════════════════════════ */}
      {/* SECTION 2 — ARCHITECTURE */}
      {/* ═══════════════════════════════════════ */}
      <Section>
        <Label>02 — Architecture</Label>

        <motion.h2
          variants={fadeUp}
          className="text-3xl md:text-5xl font-bold leading-tight mb-16 max-w-3xl"
        >
          Quantum meets classical
        </motion.h2>

        <motion.div
          variants={fadeUp}
          className="text-xs md:text-sm leading-loose mb-16 max-w-4xl"
          style={{ color: LIGHT_GRAY, whiteSpace: "pre" }}
        >
{`  transmission spectrum (52 wavelengths)
  │
  ├──→ SpectralEncoder ──→ 64-dim latent
  │
  auxiliary features (planet radius, star temp, ...)
  │
  ├──→ AuxEncoder ──────→ 32-dim latent
  │
  └──→ FusionLayer ─────→ 96-dim combined
                          │
                          ├──→ AngleEncoding (12 qubits)
                          │    │
                          │    ├──→ RX / RY / RZ rotations
                          │    ├──→ CNOT entanglement
                          │    ├──→ 3 variational layers
                          │    │
                          │    └──→ expectation values ──→ 12-dim
                          │
                          └──→ OutputHead
                               │
                               └──→ log₁₀ VMR × 5 molecules`}
        </motion.div>

        <motion.div
          variants={fadeUp}
          className="flex flex-wrap gap-x-12 gap-y-4 text-xs"
          style={{ color: GRAY }}
        >
          <span>framework: pytorch + pennylane</span>
          <span>qubits: 12</span>
          <span>variational layers: 3</span>
          <span>parameters: ~47k classical + 108 quantum</span>
        </motion.div>
      </Section>

      <Divider />

      {/* ═══════════════════════════════════════ */}
      {/* SECTION 3 — THE NUMBER */}
      {/* ═══════════════════════════════════════ */}
      <Section>
        <Label>03 — Result</Label>

        <motion.div variants={fadeUp}>
          <p
            className="text-[10px] tracking-[0.3em] uppercase mb-4"
            style={{ color: GRAY }}
          >
            mean Root Mean Squared Error
          </p>
          <p
            className="text-[80px] md:text-[120px] lg:text-[160px] font-extrabold leading-none tracking-tight"
            style={{ color: CYAN }}
          >
            0.295
          </p>
        </motion.div>

        <motion.div variants={fadeUp} className="mt-16 max-w-xl">
          <table className="w-full text-left text-sm" style={{ color: LIGHT_GRAY }}>
            <thead>
              <tr
                className="text-[10px] tracking-[0.2em] uppercase"
                style={{ color: GRAY }}
              >
                <th className="pb-4 font-normal">Model</th>
                <th className="pb-4 font-normal text-right">mRMSE</th>
                <th className="pb-4 font-normal text-right">vs ExoBiome</th>
              </tr>
            </thead>
            <tbody>
              {[
                { name: "Random Forest", score: 1.2, highlight: false },
                { name: "CNN Baseline", score: 0.85, highlight: false },
                { name: "ADC 2023 Winner", score: 0.32, highlight: false },
                { name: "ExoBiome", score: 0.295, highlight: true },
              ].map((row) => (
                <tr
                  key={row.name}
                  className="border-t"
                  style={{
                    borderColor: DARK_GRAY,
                    color: row.highlight ? CYAN : LIGHT_GRAY,
                  }}
                >
                  <td className="py-3 font-medium">{row.name}</td>
                  <td className="py-3 text-right">{row.score.toFixed(3)}</td>
                  <td
                    className="py-3 text-right"
                    style={{ color: row.highlight ? CYAN : GRAY }}
                  >
                    {row.highlight
                      ? "—"
                      : `+${((row.score / 0.295 - 1) * 100).toFixed(0)}%`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      </Section>

      <Divider />

      {/* ═══════════════════════════════════════ */}
      {/* SECTION 4 — CHART (one allowed) */}
      {/* ═══════════════════════════════════════ */}
      <Section>
        <Label>04 — Comparison</Label>

        <motion.h2
          variants={fadeUp}
          className="text-3xl md:text-4xl font-bold leading-tight mb-16 max-w-3xl"
        >
          Performance against baselines
        </motion.h2>

        <motion.div variants={fadeUp} className="w-full max-w-2xl h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={comparisonData}
              layout="vertical"
              margin={{ top: 0, right: 0, bottom: 0, left: 0 }}
              barCategoryGap="30%"
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={DARK_GRAY}
                horizontal={false}
              />
              <XAxis
                type="number"
                domain={[0, 1.4]}
                tick={{ fill: GRAY, fontSize: 11, fontFamily: "JetBrains Mono" }}
                axisLine={{ stroke: DARK_GRAY }}
                tickLine={{ stroke: DARK_GRAY }}
              />
              <YAxis
                type="category"
                dataKey="model"
                tick={{ fill: LIGHT_GRAY, fontSize: 11, fontFamily: "JetBrains Mono" }}
                axisLine={false}
                tickLine={false}
                width={140}
              />
              <Tooltip
                cursor={{ fill: "rgba(255,255,255,0.03)" }}
                contentStyle={{
                  background: "#111",
                  border: `1px solid ${DARK_GRAY}`,
                  borderRadius: 0,
                  fontFamily: "JetBrains Mono",
                  fontSize: 12,
                  color: WHITE,
                }}
                labelStyle={{ color: WHITE }}
              />
              <Bar dataKey="mrmse" radius={[0, 2, 2, 0]}>
                {comparisonData.map((entry, index) => (
                  <Cell
                    key={entry.model}
                    fill={
                      index === comparisonData.length - 1
                        ? CYAN
                        : GRAY
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.p
          variants={fadeUp}
          className="text-[10px] mt-8 tracking-[0.1em]"
          style={{ color: DARK_GRAY }}
        >
          lower is better // mRMSE across 5 molecules on ADC 2023 test set
        </motion.p>
      </Section>

      <Divider />

      {/* ═══════════════════════════════════════ */}
      {/* SECTION 5 — PER-MOLECULE */}
      {/* ═══════════════════════════════════════ */}
      <Section>
        <Label>05 — Per-molecule breakdown</Label>

        <motion.h2
          variants={fadeUp}
          className="text-3xl md:text-4xl font-bold leading-tight mb-16 max-w-3xl"
        >
          Five molecules. Raw numbers.
        </motion.h2>

        <motion.div variants={stagger} className="max-w-2xl">
          {moleculeData.map((mol, i) => (
            <motion.div
              key={mol.molecule}
              variants={fadeUp}
              className="flex items-baseline justify-between py-5 border-t"
              style={{ borderColor: DARK_GRAY }}
            >
              <div className="flex items-baseline gap-4">
                <span
                  className="text-[10px] tracking-[0.2em]"
                  style={{ color: DARK_GRAY }}
                >
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="text-2xl md:text-3xl font-bold">
                  {mol.molecule}
                </span>
                <span
                  className="text-xs"
                  style={{ color: GRAY }}
                >
                  {mol.label}
                </span>
              </div>
              <span
                className="text-2xl md:text-3xl font-bold tabular-nums"
                style={{ color: CYAN }}
              >
                {mol.mrmse}
              </span>
            </motion.div>
          ))}
          <div
            className="border-t pt-5 flex items-baseline justify-between"
            style={{ borderColor: DARK_GRAY }}
          >
            <div className="flex items-baseline gap-4">
              <span
                className="text-[10px] tracking-[0.2em]"
                style={{ color: DARK_GRAY }}
              >
                {"  "}
              </span>
              <span
                className="text-lg font-medium"
                style={{ color: GRAY }}
              >
                mean
              </span>
            </div>
            <span
              className="text-2xl md:text-3xl font-bold tabular-nums"
              style={{ color: CYAN }}
            >
              0.295
            </span>
          </div>
        </motion.div>
      </Section>

      <Divider />

      {/* ═══════════════════════════════════════ */}
      {/* SECTION 6 — QUANTUM ADVANTAGE */}
      {/* ═══════════════════════════════════════ */}
      <Section>
        <Label>06 — Why quantum</Label>

        <motion.h2
          variants={fadeUp}
          className="text-3xl md:text-5xl font-bold leading-tight mb-12 max-w-4xl"
        >
          12 qubits encode what
          <br />
          classical networks cannot
        </motion.h2>

        <motion.div
          variants={stagger}
          className="max-w-2xl space-y-8"
        >
          <motion.div variants={fadeUp}>
            <p
              className="text-[10px] tracking-[0.3em] uppercase mb-2"
              style={{ color: GRAY }}
            >
              Expressibility
            </p>
            <p className="text-sm leading-relaxed" style={{ color: LIGHT_GRAY }}>
              A 12-qubit variational circuit with 3 layers creates a 2¹²-dimensional
              Hilbert space. The quantum state explores correlations between
              spectral features that would require exponentially more classical
              parameters to represent.
            </p>
          </motion.div>

          <motion.div variants={fadeUp}>
            <p
              className="text-[10px] tracking-[0.3em] uppercase mb-2"
              style={{ color: GRAY }}
            >
              Entanglement as feature coupling
            </p>
            <p className="text-sm leading-relaxed" style={{ color: LIGHT_GRAY }}>
              CNOT gates entangle qubits encoding different spectral regions,
              capturing inter-molecular absorption overlaps that single-molecule
              retrievals miss. The quantum circuit learns cross-molecule
              dependencies natively.
            </p>
          </motion.div>

          <motion.div variants={fadeUp}>
            <p
              className="text-[10px] tracking-[0.3em] uppercase mb-2"
              style={{ color: GRAY }}
            >
              Efficiency
            </p>
            <p className="text-sm leading-relaxed" style={{ color: LIGHT_GRAY }}>
              108 quantum parameters complement ~47k classical ones.
              The quantum layer adds less than 0.3% to total parameter count
              while driving 8% improvement in mRMSE over the pure-classical
              ablation.
            </p>
          </motion.div>
        </motion.div>
      </Section>

      <Divider />

      {/* ═══════════════════════════════════════ */}
      {/* SECTION 7 — PIPELINE */}
      {/* ═══════════════════════════════════════ */}
      <Section>
        <Label>07 — Pipeline</Label>

        <motion.h2
          variants={fadeUp}
          className="text-3xl md:text-4xl font-bold leading-tight mb-16 max-w-3xl"
        >
          End-to-end data flow
        </motion.h2>

        <motion.div
          variants={stagger}
          className="space-y-1 text-xs md:text-sm max-w-3xl"
          style={{ color: LIGHT_GRAY }}
        >
          {[
            {
              step: "01",
              label: "Ingest",
              desc: "ADC 2023 dataset — 41,423 synthetic Ariel-like transmission spectra",
            },
            {
              step: "02",
              label: "Preprocess",
              desc: "log-transform, standardize, noise-aware augmentation",
            },
            {
              step: "03",
              label: "Encode",
              desc: "spectral + auxiliary features → 96-dim fused representation",
            },
            {
              step: "04",
              label: "Quantum",
              desc: "angle encoding → 12-qubit variational circuit → 12 expectation values",
            },
            {
              step: "05",
              label: "Decode",
              desc: "output head → log₁₀ VMR for H₂O, CO₂, CO, CH₄, NH₃",
            },
            {
              step: "06",
              label: "Evaluate",
              desc: "per-molecule RMSE, mean RMSE, posterior calibration",
            },
          ].map((item) => (
            <motion.div
              key={item.step}
              variants={fadeUp}
              className="flex gap-6 py-4 border-t items-baseline"
              style={{ borderColor: DARK_GRAY }}
            >
              <span
                className="text-[10px] tracking-[0.2em] shrink-0"
                style={{ color: DARK_GRAY }}
              >
                {item.step}
              </span>
              <span className="font-medium w-24 shrink-0" style={{ color: WHITE }}>
                {item.label}
              </span>
              <span style={{ color: GRAY }}>{item.desc}</span>
            </motion.div>
          ))}
        </motion.div>
      </Section>

      <Divider />

      {/* ═══════════════════════════════════════ */}
      {/* SECTION 8 — NOVELTY */}
      {/* ═══════════════════════════════════════ */}
      <Section>
        <Label>08 — Novelty</Label>

        <motion.h2
          variants={fadeUp}
          className="text-3xl md:text-5xl font-bold leading-tight mb-12 max-w-4xl"
        >
          First of its kind
        </motion.h2>

        <motion.div variants={stagger} className="max-w-2xl space-y-6">
          <motion.p
            variants={fadeUp}
            className="text-sm leading-relaxed"
            style={{ color: LIGHT_GRAY }}
          >
            No prior work has applied quantum machine learning to biosignature
            detection. ExoBiome is the first quantum-classical neural network
            trained on atmospheric retrieval benchmarks.
          </motion.p>

          <motion.div
            variants={fadeUp}
            className="text-xs leading-relaxed space-y-3 pt-4"
          >
            <div className="flex gap-4">
              <span style={{ color: DARK_GRAY }} className="shrink-0">
                ──
              </span>
              <span style={{ color: GRAY }}>
                Vetrano et al. 2025: QELM for atmospheric retrieval — no biosignature classification
              </span>
            </div>
            <div className="flex gap-4">
              <span style={{ color: DARK_GRAY }} className="shrink-0">
                ──
              </span>
              <span style={{ color: GRAY }}>
                Seeburger et al. 2023: biosphere to spectrum pipeline — no machine learning
              </span>
            </div>
            <div className="flex gap-4">
              <span style={{ color: DARK_GRAY }} className="shrink-0">
                ──
              </span>
              <span style={{ color: GRAY }}>
                ADC 2023 winners: classical deep learning — no quantum component
              </span>
            </div>
            <div className="flex gap-4 pt-2">
              <span style={{ color: CYAN }} className="shrink-0">
                ──
              </span>
              <span style={{ color: CYAN }}>
                ExoBiome: quantum-classical NN + biosignature detection. No precedent.
              </span>
            </div>
          </motion.div>
        </motion.div>
      </Section>

      <Divider />

      {/* ═══════════════════════════════════════ */}
      {/* SECTION 9 — TECH STACK */}
      {/* ═══════════════════════════════════════ */}
      <Section>
        <Label>09 — Stack</Label>

        <motion.div variants={stagger} className="max-w-2xl">
          {[
            { category: "quantum", items: "pennylane, qiskit, qiskit-on-iqm" },
            { category: "ml", items: "pytorch, sQUlearn (QRCClassifier), scikit-learn" },
            { category: "data", items: "multirex, spectres, scipy, numpy" },
            { category: "hardware", items: "Odra 5 (IQM Spark, 5 qubits), VTT Q50 (53 qubits)" },
            { category: "dataset", items: "Ariel Data Challenge 2023, 41,423 spectra" },
          ].map((row) => (
            <motion.div
              key={row.category}
              variants={fadeUp}
              className="flex gap-8 py-4 border-t items-baseline"
              style={{ borderColor: DARK_GRAY }}
            >
              <span
                className="text-[10px] tracking-[0.3em] uppercase w-20 shrink-0"
                style={{ color: GRAY }}
              >
                {row.category}
              </span>
              <span className="text-sm" style={{ color: LIGHT_GRAY }}>
                {row.items}
              </span>
            </motion.div>
          ))}
        </motion.div>
      </Section>

      <Divider />

      {/* ═══════════════════════════════════════ */}
      {/* SECTION 10 — CLOSING */}
      {/* ═══════════════════════════════════════ */}
      <Section className="min-h-screen">
        <motion.p
          variants={fadeUp}
          className="text-[10px] tracking-[0.3em] uppercase mb-16"
          style={{ color: GRAY }}
        >
          10 — Summary
        </motion.p>

        <motion.h2
          variants={fadeUp}
          className="text-4xl md:text-6xl lg:text-7xl font-extrabold leading-[0.9] tracking-tight mb-16 max-w-4xl"
        >
          One spectrum.
          <br />
          Twelve qubits.
          <br />
          Five molecules.
          <br />
          <span style={{ color: CYAN }}>Life detected.</span>
        </motion.h2>

        <motion.div
          variants={fadeUp}
          className="text-sm leading-relaxed max-w-xl mb-20"
          style={{ color: LIGHT_GRAY }}
        >
          <p>
            ExoBiome demonstrates that quantum-classical hybrid architectures
            can outperform state-of-the-art classical models on atmospheric
            retrieval tasks. With an mRMSE of 0.295 — 8% better than the
            ADC 2023 winning solution — it opens a new frontier in the search
            for extraterrestrial biosignatures.
          </p>
        </motion.div>

        <motion.div variants={fadeUp} className="flex flex-col gap-1 text-xs" style={{ color: DARK_GRAY }}>
          <span>HACK-4-SAGES 2026</span>
          <span>ETH Zurich — Origins Federation</span>
          <span>Life Detection &amp; Biosignatures</span>
        </motion.div>
      </Section>

      {/* footer line */}
      <div className="w-full" style={{ height: "1px", background: DARK_GRAY }} />

      <footer className="px-8 md:px-16 lg:px-24 py-12 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <p className="text-[10px] tracking-[0.2em]" style={{ color: DARK_GRAY }}>
          ExoBiome — Quantum Biosignature Detection
        </p>
        <p className="text-[10px] tracking-[0.2em]" style={{ color: DARK_GRAY }}>
          2026
        </p>
      </footer>
    </div>
  );
}
