"use client";

import { Work_Sans } from "next/font/google";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  LabelList,
} from "recharts";

const workSans = Work_Sans({
  subsets: ["latin"],
  variable: "--font-work",
  weight: ["200", "300", "400", "500", "600", "700", "800", "900"],
});

const BLUE = "#1a3a6e";
const LIGHT_GRAY = "#e8e8e8";
const MID_GRAY = "#b0b0b0";

function Section({
  children,
  className = "",
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <motion.section
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
      transition={{ duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.section>
  );
}

function SectionNumber({ n }: { n: string }) {
  return (
    <span
      className="absolute select-none pointer-events-none font-extralight"
      style={{
        fontSize: "clamp(160px, 20vw, 280px)",
        color: LIGHT_GRAY,
        lineHeight: 1,
        top: "-0.15em",
        left: "-0.02em",
        zIndex: 0,
        letterSpacing: "-0.04em",
      }}
    >
      {n}
    </span>
  );
}

function HorizontalBar({
  label,
  value,
  maxValue,
  highlight = false,
  delay = 0,
}: {
  label: string;
  value: number;
  maxValue: number;
  highlight?: boolean;
  delay?: number;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-40px" });
  const pct = (value / maxValue) * 100;

  return (
    <div ref={ref} className="mb-5">
      <div className="flex justify-between items-baseline mb-1.5">
        <span
          className="text-xs tracking-[0.15em] uppercase"
          style={{ color: highlight ? BLUE : "#666" }}
        >
          {label}
        </span>
        <span
          className="font-semibold tabular-nums"
          style={{
            color: highlight ? BLUE : "#999",
            fontSize: highlight ? "1.1rem" : "0.9rem",
          }}
        >
          {value.toFixed(3)}
        </span>
      </div>
      <div
        className="w-full h-2 rounded-full overflow-hidden"
        style={{ backgroundColor: "#f0f0f0" }}
      >
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: highlight ? BLUE : MID_GRAY }}
          initial={{ width: 0 }}
          animate={isInView ? { width: `${pct}%` } : { width: 0 }}
          transition={{ duration: 1, delay, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
    </div>
  );
}

function PipelineBlock({
  label,
  sub,
  delay,
  width = "w-36",
}: {
  label: string;
  sub: string;
  delay: number;
  width?: string;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-40px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, scale: 0.85 }}
      animate={
        isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.85 }
      }
      transition={{ duration: 0.5, delay, ease: [0.22, 1, 0.36, 1] }}
      className={`${width} border-2 p-4 flex flex-col justify-center`}
      style={{ borderColor: BLUE }}
    >
      <div
        className="text-[10px] tracking-[0.2em] uppercase mb-1"
        style={{ color: MID_GRAY }}
      >
        {sub}
      </div>
      <div
        className="text-sm font-bold uppercase tracking-wide"
        style={{ color: BLUE }}
      >
        {label}
      </div>
    </motion.div>
  );
}

function PipelineArrow({ delay }: { delay: number }) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-40px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, scaleX: 0 }}
      animate={
        isInView ? { opacity: 1, scaleX: 1 } : { opacity: 0, scaleX: 0 }
      }
      transition={{ duration: 0.4, delay, ease: [0.22, 1, 0.36, 1] }}
      className="flex items-center origin-left"
    >
      <div className="w-10 h-0.5" style={{ backgroundColor: BLUE }} />
      <div
        className="w-0 h-0 border-t-[5px] border-b-[5px] border-l-[8px] border-transparent"
        style={{ borderLeftColor: BLUE }}
      />
    </motion.div>
  );
}

function MoleculeCard({
  molecule,
  value,
  index,
}: {
  molecule: string;
  value: number;
  index: number;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-40px" });

  const subscripts: Record<string, string> = {
    "H₂O": "Water",
    "CO₂": "Carbon Dioxide",
    CO: "Carbon Monoxide",
    "CH₄": "Methane",
    "NH₃": "Ammonia",
  };

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
      transition={{
        duration: 0.5,
        delay: index * 0.1,
        ease: [0.22, 1, 0.36, 1],
      }}
      className="border-t-2 pt-5 pb-4"
      style={{ borderColor: BLUE }}
    >
      <div
        className="text-[10px] tracking-[0.2em] uppercase mb-2"
        style={{ color: MID_GRAY }}
      >
        {subscripts[molecule]}
      </div>
      <div
        className="text-2xl font-bold mb-1 tracking-tight"
        style={{ color: BLUE }}
      >
        {molecule}
      </div>
      <div className="text-4xl font-extralight tabular-nums" style={{ color: BLUE }}>
        {value.toFixed(3)}
      </div>
      <div
        className="text-[10px] tracking-[0.15em] uppercase mt-2"
        style={{ color: MID_GRAY }}
      >
        mRMSE
      </div>
    </motion.div>
  );
}

const moleculeData = [
  { molecule: "H₂O", value: 0.218 },
  { molecule: "CO₂", value: 0.261 },
  { molecule: "CO", value: 0.327 },
  { molecule: "CH₄", value: 0.290 },
  { molecule: "NH₃", value: 0.378 },
];

const rechartData = [
  { name: "H₂O", value: 0.218 },
  { name: "CO₂", value: 0.261 },
  { name: "CH₄", value: 0.290 },
  { name: "CO", value: 0.327 },
  { name: "NH₃", value: 0.378 },
];

export default function ExoBiomePage() {
  return (
    <div
      className={`${workSans.variable} min-h-screen relative overflow-x-hidden`}
      style={{
        fontFamily: "var(--font-work), sans-serif",
        backgroundColor: "#fff",
        color: "#1a1a1a",
      }}
    >
      {/* Vertical rule — design anchor */}
      <div
        className="fixed top-0 left-8 md:left-12 w-1 h-full z-10 pointer-events-none"
        style={{ backgroundColor: BLUE }}
      />

      {/* Subtle grid guidelines */}
      <div className="fixed inset-0 pointer-events-none z-0 opacity-[0.035]">
        <div className="h-full max-w-[1400px] mx-auto grid grid-cols-12 gap-4 px-6">
          {Array.from({ length: 12 }).map((_, i) => (
            <div key={i} className="h-full bg-current" />
          ))}
        </div>
      </div>

      {/* ===================== HERO ===================== */}
      <header className="relative min-h-screen flex flex-col justify-center pl-16 md:pl-24 pr-8 md:pr-16">
        {/* Geometric accent circle */}
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 0.06 }}
          transition={{ duration: 1.2, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
          className="absolute right-[-10vw] top-[15vh] w-[50vw] h-[50vw] rounded-full pointer-events-none"
          style={{ backgroundColor: BLUE }}
        />

        {/* Small geometric rectangle accent */}
        <motion.div
          initial={{ scaleY: 0 }}
          animate={{ scaleY: 1 }}
          transition={{ duration: 0.8, delay: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="absolute top-[18vh] right-[12vw] w-3 h-32 origin-top"
          style={{ backgroundColor: BLUE }}
        />

        <motion.div
          initial={{ opacity: 0, x: -60 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.9, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
        >
          <p
            className="text-[10px] tracking-[0.3em] uppercase mb-6"
            style={{ color: MID_GRAY }}
          >
            HACK-4-SAGES 2026 &mdash; ETH Zurich
          </p>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, x: -80 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          className="font-black uppercase leading-[0.88] tracking-[-0.03em] mb-6"
          style={{
            fontSize: "clamp(4rem, 12vw, 11rem)",
            color: BLUE,
          }}
        >
          EXOBIOME
        </motion.h1>

        <motion.div
          initial={{ opacity: 0, x: -40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, delay: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-xl"
        >
          <h2
            className="text-lg md:text-xl font-light uppercase tracking-[0.08em] mb-8 leading-relaxed"
            style={{ color: "#444" }}
          >
            Quantum Biosignature Detection
            <br />
            from Exoplanet Transmission Spectra
          </h2>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.9 }}
          className="flex items-center gap-6"
        >
          <div
            className="w-16 h-px"
            style={{ backgroundColor: BLUE }}
          />
          <p
            className="text-[10px] tracking-[0.2em] uppercase"
            style={{ color: MID_GRAY }}
          >
            Life Detection &amp; Biosignatures &mdash; March 2026
          </p>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 1.5 }}
          className="absolute bottom-12 left-16 md:left-24 flex items-center gap-3"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
            className="w-px h-8"
            style={{ backgroundColor: BLUE }}
          />
          <span
            className="text-[9px] tracking-[0.2em] uppercase"
            style={{ color: MID_GRAY }}
          >
            Scroll
          </span>
        </motion.div>
      </header>

      {/* ===================== 01 — PROBLEM ===================== */}
      <Section className="relative py-32 md:py-48 pl-16 md:pl-24 pr-8 md:pr-16">
        <SectionNumber n="01" />
        <div className="relative z-10 max-w-3xl">
          <h3
            className="text-xs tracking-[0.25em] uppercase mb-8"
            style={{ color: MID_GRAY }}
          >
            The Challenge
          </h3>
          <h2
            className="text-3xl md:text-5xl font-bold uppercase tracking-[-0.02em] leading-[1.1] mb-10"
            style={{ color: BLUE }}
          >
            DETECTING LIFE BEYOND
            <br />
            OUR SOLAR SYSTEM
          </h2>
          <div className="max-w-lg">
            <p
              className="text-base font-light leading-relaxed mb-6"
              style={{ color: "#555" }}
            >
              Next-generation space telescopes like Ariel will observe
              transmission spectra of hundreds of exoplanet atmospheres.
              Extracting molecular abundances from these spectra is a
              computationally expensive inverse problem.
            </p>
            <p
              className="text-base font-light leading-relaxed"
              style={{ color: "#555" }}
            >
              ExoBiome introduces quantum-enhanced machine learning to retrieve
              atmospheric compositions with unprecedented accuracy — detecting
              the molecular fingerprints of potential biospheres at scale.
            </p>
          </div>
          {/* Geometric accent line */}
          <div
            className="mt-12 w-24 h-px"
            style={{ backgroundColor: BLUE }}
          />
        </div>
      </Section>

      {/* ===================== 02 — ARCHITECTURE ===================== */}
      <Section className="relative py-32 md:py-48 pl-16 md:pl-24 pr-8 md:pr-16">
        <SectionNumber n="02" />
        <div className="relative z-10">
          <h3
            className="text-xs tracking-[0.25em] uppercase mb-8"
            style={{ color: MID_GRAY }}
          >
            Architecture
          </h3>
          <h2
            className="text-3xl md:text-5xl font-bold uppercase tracking-[-0.02em] leading-[1.1] mb-16"
            style={{ color: BLUE }}
          >
            QUANTUM-CLASSICAL
            <br />
            HYBRID PIPELINE
          </h2>

          {/* Pipeline diagram */}
          <div className="overflow-x-auto pb-4">
            <div className="flex items-center gap-0 min-w-[900px]">
              <PipelineBlock
                label="Spectrum"
                sub="52 bins input"
                delay={0}
                width="w-32"
              />
              <PipelineArrow delay={0.15} />
              <PipelineBlock
                label="Spectral Encoder"
                sub="Feature extraction"
                delay={0.2}
                width="w-40"
              />
              <PipelineArrow delay={0.35} />

              <div className="flex flex-col gap-2">
                <PipelineBlock
                  label="Fusion"
                  sub="Combined features"
                  delay={0.4}
                  width="w-32"
                />
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.4, delay: 0.45 }}
                  className="flex items-center gap-0"
                >
                  <div className="flex items-center">
                    <PipelineBlock
                      label="Aux Encoder"
                      sub="Stellar params"
                      delay={0.3}
                      width="w-32"
                    />
                    <div className="flex items-center">
                      <div
                        className="w-6 h-0.5"
                        style={{ backgroundColor: BLUE }}
                      />
                      <div
                        className="w-0 h-0 border-t-[4px] border-b-[4px] border-l-[6px] border-transparent"
                        style={{ borderLeftColor: BLUE }}
                      />
                    </div>
                  </div>
                </motion.div>
              </div>

              <PipelineArrow delay={0.55} />

              {/* Quantum circuit — visually distinct */}
              <motion.div
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{
                  duration: 0.6,
                  delay: 0.6,
                  ease: [0.22, 1, 0.36, 1],
                }}
                className="w-44 p-5 relative"
                style={{
                  backgroundColor: BLUE,
                }}
              >
                <div className="text-[10px] tracking-[0.2em] uppercase mb-1 text-white/50">
                  Quantum Layer
                </div>
                <div className="text-sm font-bold uppercase tracking-wide text-white">
                  QELM Circuit
                </div>
                <div className="text-[10px] tracking-[0.15em] text-white/50 mt-2">
                  12 QUBITS
                </div>
                {/* Circuit lines decoration */}
                <div className="mt-3 flex flex-col gap-[3px]">
                  {Array.from({ length: 6 }).map((_, i) => (
                    <div
                      key={i}
                      className="h-px w-full"
                      style={{ backgroundColor: "rgba(255,255,255,0.2)" }}
                    />
                  ))}
                </div>
              </motion.div>

              <PipelineArrow delay={0.75} />

              <PipelineBlock
                label="5 Molecules"
                sub="log₁₀ VMR output"
                delay={0.8}
                width="w-36"
              />
            </div>
          </div>

          {/* Architecture specs */}
          <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-8 max-w-2xl">
            {[
              { label: "Input Bins", value: "52" },
              { label: "Qubits", value: "12" },
              { label: "Parameters", value: "~85K" },
              { label: "Output", value: "5 VMRs" },
            ].map((item, i) => (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{
                  duration: 0.4,
                  delay: i * 0.1,
                  ease: [0.22, 1, 0.36, 1],
                }}
              >
                <div
                  className="text-[10px] tracking-[0.2em] uppercase mb-2"
                  style={{ color: MID_GRAY }}
                >
                  {item.label}
                </div>
                <div
                  className="text-2xl font-bold"
                  style={{ color: BLUE }}
                >
                  {item.value}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </Section>

      {/* ===================== 03 — RESULTS ===================== */}
      <Section className="relative py-32 md:py-48 pl-16 md:pl-24 pr-8 md:pr-16">
        <SectionNumber n="03" />
        <div className="relative z-10">
          <h3
            className="text-xs tracking-[0.25em] uppercase mb-8"
            style={{ color: MID_GRAY }}
          >
            Performance
          </h3>
          <h2
            className="text-3xl md:text-5xl font-bold uppercase tracking-[-0.02em] leading-[1.1] mb-4"
            style={{ color: BLUE }}
          >
            BEATING THE STATE
            <br />
            OF THE ART
          </h2>

          {/* Hero metric */}
          <div className="mb-16">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{
                duration: 0.8,
                delay: 0.2,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="inline-block"
            >
              <span
                className="font-extralight tracking-[-0.04em] block"
                style={{
                  fontSize: "clamp(5rem, 14vw, 12rem)",
                  color: BLUE,
                  lineHeight: 1,
                }}
              >
                0.295
              </span>
              <span
                className="text-[10px] tracking-[0.25em] uppercase block mt-3"
                style={{ color: MID_GRAY }}
              >
                Mean RMSE &mdash; Overall Score
              </span>
            </motion.div>
          </div>

          {/* Comparison bars */}
          <div className="max-w-lg">
            <HorizontalBar
              label="ExoBiome (Quantum)"
              value={0.295}
              maxValue={1.3}
              highlight
              delay={0}
            />
            <HorizontalBar
              label="ADC 2023 Winner"
              value={0.32}
              maxValue={1.3}
              delay={0.1}
            />
            <HorizontalBar
              label="CNN Baseline"
              value={0.85}
              maxValue={1.3}
              delay={0.2}
            />
            <HorizontalBar
              label="Random Forest"
              value={1.2}
              maxValue={1.3}
              delay={0.3}
            />
          </div>

          {/* Improvement callout */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="mt-12 flex items-center gap-4"
          >
            <div
              className="w-12 h-12 flex items-center justify-center text-white text-xs font-bold"
              style={{ backgroundColor: BLUE }}
            >
              8%
            </div>
            <p
              className="text-sm font-light"
              style={{ color: "#555" }}
            >
              Improvement over the ADC 2023 competition winner
            </p>
          </motion.div>
        </div>
      </Section>

      {/* ===================== 04 — PER-MOLECULE ===================== */}
      <Section className="relative py-32 md:py-48 pl-16 md:pl-24 pr-8 md:pr-16">
        <SectionNumber n="04" />
        <div className="relative z-10">
          <h3
            className="text-xs tracking-[0.25em] uppercase mb-8"
            style={{ color: MID_GRAY }}
          >
            Per-Molecule Breakdown
          </h3>
          <h2
            className="text-3xl md:text-5xl font-bold uppercase tracking-[-0.02em] leading-[1.1] mb-16"
            style={{ color: BLUE }}
          >
            FIVE BIOSIGNATURE
            <br />
            MOLECULES
          </h2>

          {/* Molecule cards grid */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-6 md:gap-8 mb-20">
            {moleculeData.map((m, i) => (
              <MoleculeCard
                key={m.molecule}
                molecule={m.molecule}
                value={m.value}
                index={i}
              />
            ))}
          </div>

          {/* Recharts bar chart */}
          <div className="max-w-xl">
            <div
              className="text-[10px] tracking-[0.2em] uppercase mb-6"
              style={{ color: MID_GRAY }}
            >
              mRMSE by Molecule
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={rechartData}
                  margin={{ top: 20, right: 0, left: 0, bottom: 0 }}
                  barCategoryGap="30%"
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#eee"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="name"
                    axisLine={false}
                    tickLine={false}
                    tick={{
                      fontSize: 11,
                      fill: "#888",
                      fontFamily: "var(--font-work)",
                    }}
                  />
                  <YAxis
                    domain={[0, 0.45]}
                    axisLine={false}
                    tickLine={false}
                    tick={{
                      fontSize: 10,
                      fill: "#bbb",
                      fontFamily: "var(--font-work)",
                    }}
                    width={35}
                  />
                  <Bar dataKey="value" radius={[2, 2, 0, 0]}>
                    {rechartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={index === 0 ? BLUE : `${BLUE}${index === 4 ? "88" : "bb"}`}
                      />
                    ))}
                    <LabelList
                      dataKey="value"
                      position="top"
                      formatter={(v) => Number(v).toFixed(3)}
                      style={{
                        fontSize: 10,
                        fill: "#888",
                        fontFamily: "var(--font-work)",
                      }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </Section>

      {/* ===================== 05 — QUANTUM ADVANTAGE ===================== */}
      <Section className="relative py-32 md:py-48 pl-16 md:pl-24 pr-8 md:pr-16">
        <SectionNumber n="05" />
        <div className="relative z-10">
          <h3
            className="text-xs tracking-[0.25em] uppercase mb-8"
            style={{ color: MID_GRAY }}
          >
            Why Quantum
          </h3>
          <h2
            className="text-3xl md:text-5xl font-bold uppercase tracking-[-0.02em] leading-[1.1] mb-16"
            style={{ color: BLUE }}
          >
            THE QUANTUM
            <br />
            ADVANTAGE
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 max-w-4xl">
            {[
              {
                title: "Expressive Feature Maps",
                text: "Quantum circuits encode input features into an exponentially large Hilbert space, capturing complex non-linear correlations that classical models miss.",
              },
              {
                title: "Hardware Ready",
                text: "Designed to run on real quantum hardware — IQM Spark (5 qubits) and VTT Helmi (20 qubits) via PWR Wroclaw. Not just simulation.",
              },
              {
                title: "First of Its Kind",
                text: "No prior work has applied quantum machine learning to biosignature detection. ExoBiome is the first quantum-enhanced atmospheric retrieval system.",
              },
            ].map((item, i) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{
                  duration: 0.5,
                  delay: i * 0.15,
                  ease: [0.22, 1, 0.36, 1],
                }}
              >
                <div
                  className="w-8 h-1 mb-6"
                  style={{ backgroundColor: BLUE }}
                />
                <h4
                  className="text-sm font-bold uppercase tracking-[0.08em] mb-4"
                  style={{ color: BLUE }}
                >
                  {item.title}
                </h4>
                <p
                  className="text-sm font-light leading-relaxed"
                  style={{ color: "#555" }}
                >
                  {item.text}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </Section>

      {/* ===================== 06 — SUMMARY ===================== */}
      <Section className="relative py-32 md:py-48 pl-16 md:pl-24 pr-8 md:pr-16">
        <SectionNumber n="06" />
        <div className="relative z-10">
          <h3
            className="text-xs tracking-[0.25em] uppercase mb-8"
            style={{ color: MID_GRAY }}
          >
            At a Glance
          </h3>
          <h2
            className="text-3xl md:text-5xl font-bold uppercase tracking-[-0.02em] leading-[1.1] mb-20"
            style={{ color: BLUE }}
          >
            KEY NUMBERS
          </h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-x-12 gap-y-16 max-w-4xl">
            {[
              { value: "0.295", label: "mRMSE Score", sub: "Overall" },
              { value: "8%", label: "Improvement", sub: "vs ADC Winner" },
              { value: "12", label: "Qubits", sub: "Quantum Circuit" },
              { value: "5", label: "Molecules", sub: "Detected" },
              { value: "52", label: "Spectral Bins", sub: "Input Features" },
              { value: "~85K", label: "Parameters", sub: "Total Model" },
              { value: "41K", label: "Training Spectra", sub: "Dataset" },
              { value: "1st", label: "Quantum ML", sub: "For Biosignatures" },
            ].map((item, i) => (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{
                  duration: 0.4,
                  delay: i * 0.08,
                  ease: [0.22, 1, 0.36, 1],
                }}
              >
                <div
                  className="text-4xl md:text-5xl font-extralight tracking-tight mb-2"
                  style={{ color: BLUE }}
                >
                  {item.value}
                </div>
                <div
                  className="text-xs font-semibold uppercase tracking-[0.1em] mb-1"
                  style={{ color: "#333" }}
                >
                  {item.label}
                </div>
                <div
                  className="text-[10px] tracking-[0.15em] uppercase"
                  style={{ color: MID_GRAY }}
                >
                  {item.sub}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </Section>

      {/* ===================== FOOTER ===================== */}
      <footer className="relative py-24 pl-16 md:pl-24 pr-8 md:pr-16 border-t" style={{ borderColor: LIGHT_GRAY }}>
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-8">
          <div>
            <div
              className="text-2xl font-black uppercase tracking-[-0.02em] mb-2"
              style={{ color: BLUE }}
            >
              EXOBIOME
            </div>
            <div
              className="text-[10px] tracking-[0.2em] uppercase"
              style={{ color: MID_GRAY }}
            >
              Quantum Biosignature Detection
            </div>
          </div>
          <div className="text-right">
            <div
              className="text-[10px] tracking-[0.2em] uppercase mb-1"
              style={{ color: MID_GRAY }}
            >
              HACK-4-SAGES 2026
            </div>
            <div
              className="text-[10px] tracking-[0.2em] uppercase"
              style={{ color: MID_GRAY }}
            >
              ETH Zurich &mdash; Life Detection &amp; Biosignatures
            </div>
          </div>
        </div>

        {/* Bottom geometric accent */}
        <motion.div
          initial={{ scaleX: 0 }}
          whileInView={{ scaleX: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
          className="mt-16 h-px origin-left"
          style={{ backgroundColor: BLUE }}
        />
      </footer>
    </div>
  );
}
