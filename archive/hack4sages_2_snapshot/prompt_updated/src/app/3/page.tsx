"use client";

import { useEffect, useRef, useState, ReactNode } from "react";
import { motion, useInView, AnimatePresence } from "framer-motion";
import { Cormorant_Garamond, JetBrains_Mono } from "next/font/google";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const cormorant = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  style: ["normal", "italic"],
  variable: "--font-cormorant",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["300", "400", "500", "700"],
  variable: "--font-jetbrains",
});

// ─── Grain Overlay ───────────────────────────────────────────────────────────

function GrainOverlay() {
  return (
    <div
      className="pointer-events-none fixed inset-0 z-[9999] opacity-[0.035]"
      style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        backgroundRepeat: "repeat",
        backgroundSize: "128px 128px",
      }}
    />
  );
}

// ─── Scroll Progress Bar ─────────────────────────────────────────────────────

function ScrollProgress() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const onScroll = () => {
      const scrollTop = window.scrollY;
      const docHeight =
        document.documentElement.scrollHeight - window.innerHeight;
      setProgress(docHeight > 0 ? scrollTop / docHeight : 0);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="fixed top-0 left-0 z-[9998] h-[2px] w-full">
      <motion.div
        className="h-full origin-left"
        style={{
          scaleX: progress,
          background:
            "linear-gradient(90deg, #c9a96e 0%, #f5f0e8 50%, #c9a96e 100%)",
        }}
      />
    </div>
  );
}

// ─── Section Reveal Wrapper ──────────────────────────────────────────────────

function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
      transition={{ duration: 0.9, delay, ease: [0.22, 1, 0.36, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// ─── Slide Navigation Dots ───────────────────────────────────────────────────

const SECTION_IDS = [
  "hero",
  "problem",
  "approach",
  "architecture",
  "results",
  "molecules",
  "dataset",
  "team",
];

const SECTION_LABELS = [
  "I",
  "II",
  "III",
  "IV",
  "V",
  "VI",
  "VII",
  "VIII",
];

function SlideNav() {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const idx = SECTION_IDS.indexOf(entry.target.id);
            if (idx !== -1) setActive(idx);
          }
        });
      },
      { threshold: 0.35 }
    );

    SECTION_IDS.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <nav className="fixed right-6 bottom-8 z-[9997] flex items-center gap-3">
      {SECTION_IDS.map((id, i) => (
        <button
          key={id}
          onClick={() =>
            document.getElementById(id)?.scrollIntoView({ behavior: "smooth" })
          }
          className="group relative flex flex-col items-center"
        >
          <span
            className={`${jetbrains.className} mb-1 text-[9px] tracking-widest transition-all duration-300 ${
              active === i
                ? "text-[#c9a96e] opacity-100"
                : "text-[#f5f0e8] opacity-0 group-hover:opacity-40"
            }`}
          >
            {SECTION_LABELS[i]}
          </span>
          <span
            className={`block rounded-full transition-all duration-500 ${
              active === i
                ? "h-2 w-2 bg-[#c9a96e]"
                : "h-1.5 w-1.5 bg-[#f5f0e8]/20 group-hover:bg-[#f5f0e8]/40"
            }`}
          />
        </button>
      ))}
    </nav>
  );
}

// ─── Section Number ──────────────────────────────────────────────────────────

function SectionNumber({ n }: { n: string }) {
  return (
    <span
      className={`${cormorant.className} pointer-events-none absolute -top-8 left-0 select-none text-[12rem] font-light leading-none text-[#f5f0e8]/[0.03] md:text-[16rem]`}
    >
      {n}
    </span>
  );
}

// ─── Horizontal Rule ─────────────────────────────────────────────────────────

function Rule() {
  return <div className="mx-auto my-6 h-px w-16 bg-[#c9a96e]/30" />;
}

// ─── Molecule Data ───────────────────────────────────────────────────────────

const moleculeData = [
  { molecule: "H₂O", mrmse: 0.218, color: "#5b98d4" },
  { molecule: "CO₂", mrmse: 0.261, color: "#e07a5f" },
  { molecule: "CO", mrmse: 0.327, color: "#81b29a" },
  { molecule: "CH₄", mrmse: 0.29, color: "#f2cc8f" },
  { molecule: "NH₃", mrmse: 0.378, color: "#c98bb9" },
];

const comparisonData = [
  { model: "ExoBiome (Ours)", mrmse: 0.295 },
  { model: "ADC 2023 Winner", mrmse: 0.32 },
  { model: "CNN Baseline", mrmse: 0.85 },
  { model: "Random Forest", mrmse: 1.2 },
];

// ─── Custom Tooltip for Recharts ─────────────────────────────────────────────

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded border border-[#c9a96e]/20 bg-[#141414] px-3 py-2">
      <p className={`${jetbrains.className} text-xs text-[#f5f0e8]/60`}>
        {label}
      </p>
      <p className={`${jetbrains.className} text-sm text-[#c9a96e]`}>
        mRMSE: {payload[0].value.toFixed(3)}
      </p>
    </div>
  );
}

// ─── Architecture Flow ───────────────────────────────────────────────────────

function ArchitectureFlow() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  const steps = [
    {
      label: "Transmission Spectrum",
      detail: "52 wavelength bins",
      side: "input",
    },
    {
      label: "Auxiliary Features",
      detail: "Planet radius, mass, T★, log g",
      side: "input",
    },
    { label: "Spectral Encoder", detail: "Conv1D → BatchNorm → ReLU", side: "encoder" },
    { label: "Aux Encoder", detail: "Linear → LayerNorm → GELU", side: "encoder" },
    { label: "Fusion Layer", detail: "Concatenation + Projection", side: "fusion" },
    {
      label: "Quantum Circuit",
      detail: "12 qubits · angle encoding · entangling layers",
      side: "quantum",
    },
    {
      label: "Measurement → Readout",
      detail: "⟨Z⟩ expectations · linear head",
      side: "quantum",
    },
    {
      label: "log₁₀ VMR Output",
      detail: "H₂O · CO₂ · CO · CH₄ · NH₃",
      side: "output",
    },
  ];

  const sideColors: Record<string, string> = {
    input: "#5b98d4",
    encoder: "#81b29a",
    fusion: "#f2cc8f",
    quantum: "#c9a96e",
    output: "#e07a5f",
  };

  return (
    <div ref={ref} className="relative mx-auto max-w-xl">
      {steps.map((step, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, x: -30 }}
          animate={isInView ? { opacity: 1, x: 0 } : {}}
          transition={{
            duration: 0.7,
            delay: i * 0.12,
            ease: [0.22, 1, 0.36, 1],
          }}
          className="relative flex items-start gap-6 pb-10"
        >
          {/* vertical line */}
          {i < steps.length - 1 && (
            <div
              className="absolute left-[7px] top-5 h-full w-px"
              style={{ background: `${sideColors[step.side]}33` }}
            />
          )}
          {/* dot */}
          <div
            className="relative z-10 mt-1.5 h-[15px] w-[15px] flex-shrink-0 rounded-full border-2"
            style={{
              borderColor: sideColors[step.side],
              background:
                step.side === "quantum" ? sideColors[step.side] : "transparent",
            }}
          />
          {/* text */}
          <div>
            <p
              className={`${cormorant.className} text-xl font-medium text-[#f5f0e8] md:text-2xl`}
            >
              {step.label}
            </p>
            <p
              className={`${jetbrains.className} mt-1 text-xs tracking-wide text-[#f5f0e8]/40`}
            >
              {step.detail}
            </p>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

// ─── Animated Counter ────────────────────────────────────────────────────────

function AnimatedNumber({
  value,
  decimals = 3,
}: {
  value: number;
  decimals?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true });
  const [display, setDisplay] = useState("0.000");

  useEffect(() => {
    if (!isInView) return;
    const duration = 1800;
    const start = performance.now();

    const animate = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 4);
      const current = eased * value;
      setDisplay(current.toFixed(decimals));
      if (progress < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [isInView, value, decimals]);

  return <span ref={ref}>{display}</span>;
}

// ─── Molecule Table ──────────────────────────────────────────────────────────

function MoleculeTable() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-60px" });

  return (
    <div ref={ref} className="mx-auto max-w-lg">
      <div className="overflow-hidden rounded-sm border border-[#f5f0e8]/[0.06]">
        {/* header */}
        <div className="grid grid-cols-3 border-b border-[#f5f0e8]/[0.06] bg-[#f5f0e8]/[0.02] px-6 py-3">
          <span
            className={`${jetbrains.className} text-[10px] uppercase tracking-[0.2em] text-[#f5f0e8]/30`}
          >
            Molecule
          </span>
          <span
            className={`${jetbrains.className} text-center text-[10px] uppercase tracking-[0.2em] text-[#f5f0e8]/30`}
          >
            mRMSE
          </span>
          <span
            className={`${jetbrains.className} text-right text-[10px] uppercase tracking-[0.2em] text-[#f5f0e8]/30`}
          >
            Performance
          </span>
        </div>
        {/* rows */}
        {moleculeData.map((mol, i) => (
          <motion.div
            key={mol.molecule}
            initial={{ opacity: 0, x: -20 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.6, delay: i * 0.1 }}
            className={`grid grid-cols-3 items-center px-6 py-4 ${
              i % 2 === 0 ? "bg-transparent" : "bg-[#f5f0e8]/[0.015]"
            } ${
              i < moleculeData.length - 1
                ? "border-b border-[#f5f0e8]/[0.04]"
                : ""
            }`}
          >
            <span
              className={`${cormorant.className} text-lg font-medium`}
              style={{ color: mol.color }}
            >
              {mol.molecule}
            </span>
            <span
              className={`${jetbrains.className} text-center text-sm text-[#f5f0e8]/80`}
            >
              {mol.mrmse.toFixed(3)}
            </span>
            <div className="flex justify-end">
              <div className="h-1.5 w-24 overflow-hidden rounded-full bg-[#f5f0e8]/[0.06]">
                <motion.div
                  initial={{ width: 0 }}
                  animate={
                    isInView
                      ? { width: `${((1 - mol.mrmse) / 1) * 100}%` }
                      : {}
                  }
                  transition={{ duration: 1.2, delay: 0.3 + i * 0.1 }}
                  className="h-full rounded-full"
                  style={{ background: mol.color }}
                />
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAGE
// ═══════════════════════════════════════════════════════════════════════════════

export default function ExoBiomePage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div
      className={`${cormorant.variable} ${jetbrains.variable} relative min-h-screen bg-[#0a0a0a] text-[#f5f0e8]`}
    >
      <GrainOverlay />
      <ScrollProgress />
      {mounted && <SlideNav />}

      {/* ═══ SECTION 01 — HERO ══════════════════════════════════════════════ */}
      <section
        id="hero"
        className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6"
      >
        <SectionNumber n="01" />

        {/* Floating particles */}
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          {[...Array(6)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute rounded-full bg-[#c9a96e]"
              style={{
                width: 2 + Math.random() * 3,
                height: 2 + Math.random() * 3,
                left: `${15 + Math.random() * 70}%`,
                top: `${15 + Math.random() * 70}%`,
                opacity: 0.08 + Math.random() * 0.08,
              }}
              animate={{
                y: [0, -30 - Math.random() * 40, 0],
                x: [0, 10 - Math.random() * 20, 0],
                opacity: [0.06, 0.15, 0.06],
              }}
              transition={{
                duration: 8 + Math.random() * 6,
                repeat: Infinity,
                ease: "easeInOut",
                delay: Math.random() * 4,
              }}
            />
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 2, ease: "easeOut" }}
          className="relative z-10 text-center"
        >
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.3 }}
            className={`${jetbrains.className} mb-8 text-[10px] uppercase tracking-[0.4em] text-[#c9a96e]/70`}
          >
            HACK-4-SAGES 2026 &middot; ETH Zurich
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.2, delay: 0.6 }}
            className={`${cormorant.className} text-6xl font-light leading-[0.95] tracking-tight md:text-8xl lg:text-9xl`}
          >
            Exo
            <span className="italic text-[#c9a96e]">Biome</span>
          </motion.h1>

          <motion.div
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 1, delay: 1.2 }}
            className="mx-auto my-8 h-px w-24 bg-gradient-to-r from-transparent via-[#c9a96e]/40 to-transparent"
          />

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1.2, delay: 1.4 }}
            className={`${cormorant.className} mx-auto max-w-xl text-xl font-light italic leading-relaxed text-[#f5f0e8]/50 md:text-2xl`}
          >
            Quantum biosignature detection
            <br />
            from exoplanet transmission spectra
          </motion.p>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 2 }}
            className={`${jetbrains.className} mt-12 text-[10px] uppercase tracking-[0.3em] text-[#f5f0e8]/20`}
          >
            Scroll to explore
          </motion.p>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 2.2 }}
            className="mx-auto mt-4 flex flex-col items-center"
          >
            <motion.div
              animate={{ y: [0, 6, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="h-8 w-px bg-gradient-to-b from-[#c9a96e]/40 to-transparent"
            />
          </motion.div>
        </motion.div>
      </section>

      {/* ═══ SECTION 02 — THE PROBLEM ══════════════════════════════════════ */}
      <section
        id="problem"
        className="relative min-h-screen px-6 py-32 md:py-40"
      >
        <SectionNumber n="02" />
        <div className="mx-auto max-w-3xl">
          <Reveal>
            <p
              className={`${jetbrains.className} mb-4 text-[10px] uppercase tracking-[0.3em] text-[#c9a96e]/60`}
            >
              The Problem
            </p>
          </Reveal>

          <Reveal delay={0.15}>
            <h2
              className={`${cormorant.className} text-4xl font-light leading-[1.15] md:text-5xl lg:text-6xl`}
            >
              Telescopes capture light filtered through
              <span className="italic text-[#c9a96e]"> alien atmospheres</span>.
              <br />
              Hidden in the noise are the fingerprints of{" "}
              <span className="italic text-[#c9a96e]">life</span>.
            </h2>
          </Reveal>

          <Reveal delay={0.3}>
            <Rule />
          </Reveal>

          <Reveal delay={0.4}>
            <p
              className={`${cormorant.className} mx-auto max-w-xl text-center text-lg font-light leading-relaxed text-[#f5f0e8]/45 md:text-xl`}
            >
              Atmospheric retrieval &mdash; inferring molecular abundances from
              transmission spectra &mdash; is a degenerate inverse problem.
              Classical methods struggle with high-dimensional parameter spaces
              and multi-modal posteriors. We asked: can quantum computing help?
            </p>
          </Reveal>

          <Reveal delay={0.55}>
            <div className="mt-16 grid grid-cols-1 gap-8 md:grid-cols-3">
              {[
                {
                  number: "10⁻¹²",
                  label: "Detection threshold",
                  desc: "Volume mixing ratios span 12 orders of magnitude",
                },
                {
                  number: "5",
                  label: "Target molecules",
                  desc: "H₂O, CO₂, CO, CH₄, NH₃ — key biosignature gases",
                },
                {
                  number: "52",
                  label: "Wavelength bins",
                  desc: "Ariel-class spectral resolution across 0.5–7.8 µm",
                },
              ].map((item, i) => (
                <div key={i} className="text-center">
                  <p
                    className={`${cormorant.className} text-4xl font-light italic text-[#c9a96e] md:text-5xl`}
                  >
                    {item.number}
                  </p>
                  <p
                    className={`${jetbrains.className} mt-2 text-[10px] uppercase tracking-[0.2em] text-[#f5f0e8]/50`}
                  >
                    {item.label}
                  </p>
                  <p
                    className={`${cormorant.className} mt-2 text-sm font-light text-[#f5f0e8]/30`}
                  >
                    {item.desc}
                  </p>
                </div>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═══ SECTION 03 — APPROACH ═════════════════════════════════════════ */}
      <section
        id="approach"
        className="relative min-h-screen px-6 py-32 md:py-40"
      >
        <SectionNumber n="03" />
        <div className="mx-auto max-w-3xl">
          <Reveal>
            <p
              className={`${jetbrains.className} mb-4 text-[10px] uppercase tracking-[0.3em] text-[#c9a96e]/60`}
            >
              Our Approach
            </p>
          </Reveal>

          <Reveal delay={0.15}>
            <h2
              className={`${cormorant.className} text-4xl font-light leading-[1.15] md:text-5xl lg:text-6xl`}
            >
              A <span className="italic text-[#c9a96e]">quantum-classical</span>{" "}
              neural network trained on{" "}
              <span className="italic">41,423 synthetic spectra</span>
            </h2>
          </Reveal>

          <Reveal delay={0.3}>
            <Rule />
          </Reveal>

          <Reveal delay={0.4}>
            <div className="mt-10 space-y-12">
              {[
                {
                  step: "01",
                  title: "Encode",
                  body: "Classical neural encoders compress the 52-channel spectrum and auxiliary planetary parameters into a compact latent representation.",
                },
                {
                  step: "02",
                  title: "Fuse",
                  body: "Spectral and auxiliary embeddings are concatenated and projected into a unified feature vector that captures cross-modal correlations.",
                },
                {
                  step: "03",
                  title: "Quantize",
                  body: "The fused features are angle-encoded into a 12-qubit variational quantum circuit with trainable entangling layers, exploiting quantum expressibility.",
                },
                {
                  step: "04",
                  title: "Predict",
                  body: "Pauli-Z expectations from the quantum circuit feed a linear readout head, producing log₁₀ VMR predictions for five target molecules.",
                },
              ].map((item, i) => (
                <Reveal key={i} delay={0.1 * i}>
                  <div className="flex gap-6">
                    <span
                      className={`${cormorant.className} flex-shrink-0 text-3xl font-light italic text-[#c9a96e]/30`}
                    >
                      {item.step}
                    </span>
                    <div>
                      <h3
                        className={`${cormorant.className} text-2xl font-medium md:text-3xl`}
                      >
                        {item.title}
                      </h3>
                      <p
                        className={`${cormorant.className} mt-2 text-base font-light leading-relaxed text-[#f5f0e8]/45`}
                      >
                        {item.body}
                      </p>
                    </div>
                  </div>
                </Reveal>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═══ SECTION 04 — ARCHITECTURE ═════════════════════════════════════ */}
      <section
        id="architecture"
        className="relative min-h-screen px-6 py-32 md:py-40"
      >
        <SectionNumber n="04" />
        <div className="mx-auto max-w-3xl">
          <Reveal>
            <p
              className={`${jetbrains.className} mb-4 text-[10px] uppercase tracking-[0.3em] text-[#c9a96e]/60`}
            >
              Architecture
            </p>
          </Reveal>

          <Reveal delay={0.15}>
            <h2
              className={`${cormorant.className} mb-16 text-4xl font-light leading-[1.15] md:text-5xl`}
            >
              From photons to{" "}
              <span className="italic text-[#c9a96e]">
                molecular abundances
              </span>
            </h2>
          </Reveal>

          <Reveal delay={0.3}>
            <ArchitectureFlow />
          </Reveal>

          <Reveal delay={0.5}>
            <div className="mt-16 flex flex-wrap justify-center gap-x-10 gap-y-4">
              {[
                { label: "Input", color: "#5b98d4" },
                { label: "Encoder", color: "#81b29a" },
                { label: "Fusion", color: "#f2cc8f" },
                { label: "Quantum", color: "#c9a96e" },
                { label: "Output", color: "#e07a5f" },
              ].map((leg) => (
                <div key={leg.label} className="flex items-center gap-2">
                  <div
                    className="h-2 w-2 rounded-full"
                    style={{ background: leg.color }}
                  />
                  <span
                    className={`${jetbrains.className} text-[10px] uppercase tracking-[0.15em] text-[#f5f0e8]/40`}
                  >
                    {leg.label}
                  </span>
                </div>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═══ SECTION 05 — RESULTS ══════════════════════════════════════════ */}
      <section
        id="results"
        className="relative flex min-h-screen flex-col items-center justify-center px-6 py-32"
      >
        <SectionNumber n="05" />
        <div className="mx-auto w-full max-w-4xl text-center">
          <Reveal>
            <p
              className={`${jetbrains.className} mb-4 text-[10px] uppercase tracking-[0.3em] text-[#c9a96e]/60`}
            >
              Results
            </p>
          </Reveal>

          <Reveal delay={0.15}>
            <h2
              className={`${cormorant.className} mb-4 text-4xl font-light md:text-5xl`}
            >
              Mean Relative RMSE
            </h2>
          </Reveal>

          <Reveal delay={0.3}>
            <p
              className={`${cormorant.className} text-[7rem] font-light italic leading-none tracking-tight text-[#c9a96e] md:text-[9rem]`}
            >
              <AnimatedNumber value={0.295} />
            </p>
            <p
              className={`${jetbrains.className} mt-4 text-xs tracking-wide text-[#f5f0e8]/30`}
            >
              lower is better
            </p>
          </Reveal>

          <Reveal delay={0.5}>
            <div className="mx-auto mt-16 max-w-2xl">
              <div className="overflow-hidden rounded-sm border border-[#f5f0e8]/[0.06]">
                <div className="grid grid-cols-2 border-b border-[#f5f0e8]/[0.06] bg-[#f5f0e8]/[0.02] px-6 py-3">
                  <span
                    className={`${jetbrains.className} text-left text-[10px] uppercase tracking-[0.2em] text-[#f5f0e8]/30`}
                  >
                    Model
                  </span>
                  <span
                    className={`${jetbrains.className} text-right text-[10px] uppercase tracking-[0.2em] text-[#f5f0e8]/30`}
                  >
                    mRMSE
                  </span>
                </div>
                {comparisonData.map((row, i) => (
                  <div
                    key={row.model}
                    className={`grid grid-cols-2 items-center px-6 py-4 ${
                      i % 2 === 0 ? "bg-transparent" : "bg-[#f5f0e8]/[0.015]"
                    } ${
                      i < comparisonData.length - 1
                        ? "border-b border-[#f5f0e8]/[0.04]"
                        : ""
                    }`}
                  >
                    <span
                      className={`${cormorant.className} text-left text-lg ${
                        i === 0
                          ? "font-semibold text-[#c9a96e]"
                          : "font-light text-[#f5f0e8]/60"
                      }`}
                    >
                      {row.model}
                    </span>
                    <span
                      className={`${jetbrains.className} text-right text-sm ${
                        i === 0 ? "text-[#c9a96e]" : "text-[#f5f0e8]/50"
                      }`}
                    >
                      {row.mrmse.toFixed(i === 0 ? 3 : 2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </Reveal>

          <Reveal delay={0.7}>
            <div className="mx-auto mt-16 h-64 w-full max-w-xl">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={comparisonData}
                  margin={{ top: 10, right: 10, bottom: 10, left: 10 }}
                  barCategoryGap="30%"
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#f5f0e8"
                    strokeOpacity={0.04}
                    vertical={false}
                  />
                  <XAxis
                    dataKey="model"
                    tick={{
                      fill: "#f5f0e8",
                      fillOpacity: 0.3,
                      fontSize: 10,
                      fontFamily: jetbrains.style.fontFamily,
                    }}
                    axisLine={{ stroke: "#f5f0e8", strokeOpacity: 0.06 }}
                    tickLine={false}
                    interval={0}
                  />
                  <YAxis
                    tick={{
                      fill: "#f5f0e8",
                      fillOpacity: 0.3,
                      fontSize: 10,
                      fontFamily: jetbrains.style.fontFamily,
                    }}
                    axisLine={false}
                    tickLine={false}
                    domain={[0, 1.4]}
                  />
                  <Tooltip
                    content={<CustomTooltip />}
                    cursor={{ fill: "#f5f0e8", fillOpacity: 0.02 }}
                  />
                  <Bar dataKey="mrmse" radius={[3, 3, 0, 0]}>
                    {comparisonData.map((entry, index) => (
                      <Cell
                        key={index}
                        fill={index === 0 ? "#c9a96e" : "#f5f0e8"}
                        fillOpacity={index === 0 ? 0.85 : 0.12}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═══ SECTION 06 — PER-MOLECULE ═════════════════════════════════════ */}
      <section
        id="molecules"
        className="relative min-h-screen px-6 py-32 md:py-40"
      >
        <SectionNumber n="06" />
        <div className="mx-auto max-w-3xl">
          <Reveal>
            <p
              className={`${jetbrains.className} mb-4 text-[10px] uppercase tracking-[0.3em] text-[#c9a96e]/60`}
            >
              Per-Molecule Breakdown
            </p>
          </Reveal>

          <Reveal delay={0.15}>
            <h2
              className={`${cormorant.className} mb-16 text-4xl font-light leading-[1.15] md:text-5xl`}
            >
              Five molecules.{" "}
              <span className="italic text-[#c9a96e]">
                Five fingerprints of life.
              </span>
            </h2>
          </Reveal>

          <Reveal delay={0.3}>
            <MoleculeTable />
          </Reveal>

          <Reveal delay={0.5}>
            <div className="mx-auto mt-16 h-72 w-full max-w-lg">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={moleculeData}
                  margin={{ top: 10, right: 10, bottom: 10, left: 10 }}
                  barCategoryGap="25%"
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#f5f0e8"
                    strokeOpacity={0.04}
                    vertical={false}
                  />
                  <XAxis
                    dataKey="molecule"
                    tick={{
                      fill: "#f5f0e8",
                      fillOpacity: 0.5,
                      fontSize: 12,
                      fontFamily: cormorant.style.fontFamily,
                    }}
                    axisLine={{ stroke: "#f5f0e8", strokeOpacity: 0.06 }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{
                      fill: "#f5f0e8",
                      fillOpacity: 0.3,
                      fontSize: 10,
                      fontFamily: jetbrains.style.fontFamily,
                    }}
                    axisLine={false}
                    tickLine={false}
                    domain={[0, 0.5]}
                  />
                  <Tooltip
                    content={<CustomTooltip />}
                    cursor={{ fill: "#f5f0e8", fillOpacity: 0.02 }}
                  />
                  <Bar dataKey="mrmse" radius={[3, 3, 0, 0]}>
                    {moleculeData.map((entry, index) => (
                      <Cell key={index} fill={entry.color} fillOpacity={0.75} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═══ SECTION 07 — DATASET ══════════════════════════════════════════ */}
      <section
        id="dataset"
        className="relative min-h-screen px-6 py-32 md:py-40"
      >
        <SectionNumber n="07" />
        <div className="mx-auto max-w-3xl">
          <Reveal>
            <p
              className={`${jetbrains.className} mb-4 text-[10px] uppercase tracking-[0.3em] text-[#c9a96e]/60`}
            >
              Data &amp; Methodology
            </p>
          </Reveal>

          <Reveal delay={0.15}>
            <h2
              className={`${cormorant.className} mb-16 text-4xl font-light leading-[1.15] md:text-5xl`}
            >
              Trained on the{" "}
              <span className="italic text-[#c9a96e]">
                Ariel Data Challenge 2023
              </span>
            </h2>
          </Reveal>

          <Reveal delay={0.3}>
            <div className="grid grid-cols-1 gap-px overflow-hidden rounded-sm border border-[#f5f0e8]/[0.06] bg-[#f5f0e8]/[0.04] md:grid-cols-2">
              {[
                {
                  label: "Training Spectra",
                  value: "41,423",
                  desc: "Synthetic transmission spectra generated with TauREx 3",
                },
                {
                  label: "Spectral Range",
                  value: "0.5–7.8 µm",
                  desc: "52 wavelength bins matching Ariel Tier 2 resolution",
                },
                {
                  label: "Quantum Hardware",
                  value: "12 Qubits",
                  desc: "Variational circuit with angle encoding and entangling layers",
                },
                {
                  label: "Training Regime",
                  value: "Hybrid",
                  desc: "Parameter-shift gradients for quantum, backprop for classical",
                },
              ].map((item, i) => (
                <div
                  key={i}
                  className="bg-[#0a0a0a] px-8 py-8"
                >
                  <p
                    className={`${jetbrains.className} text-[10px] uppercase tracking-[0.2em] text-[#f5f0e8]/30`}
                  >
                    {item.label}
                  </p>
                  <p
                    className={`${cormorant.className} mt-2 text-3xl font-light italic text-[#c9a96e] md:text-4xl`}
                  >
                    {item.value}
                  </p>
                  <p
                    className={`${cormorant.className} mt-2 text-sm font-light text-[#f5f0e8]/35`}
                  >
                    {item.desc}
                  </p>
                </div>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═══ SECTION 08 — TEAM / CLOSING ═══════════════════════════════════ */}
      <section
        id="team"
        className="relative flex min-h-screen flex-col items-center justify-center px-6 py-32"
      >
        <SectionNumber n="08" />
        <div className="relative z-10 mx-auto max-w-3xl text-center">
          <Reveal>
            <p
              className={`${jetbrains.className} mb-4 text-[10px] uppercase tracking-[0.3em] text-[#c9a96e]/60`}
            >
              The Team
            </p>
          </Reveal>

          <Reveal delay={0.15}>
            <h2
              className={`${cormorant.className} text-4xl font-light leading-[1.15] md:text-5xl lg:text-6xl`}
            >
              Built with{" "}
              <span className="italic text-[#c9a96e]">curiosity</span>,
              <br />
              powered by{" "}
              <span className="italic text-[#c9a96e]">quantum mechanics</span>
            </h2>
          </Reveal>

          <Reveal delay={0.3}>
            <Rule />
          </Reveal>

          <Reveal delay={0.4}>
            <p
              className={`${cormorant.className} mx-auto max-w-lg text-lg font-light italic leading-relaxed text-[#f5f0e8]/40 md:text-xl`}
            >
              ExoBiome is the first application of quantum machine learning to
              biosignature detection &mdash; bridging quantum computing and
              astrobiology to search for life beyond Earth.
            </p>
          </Reveal>

          <Reveal delay={0.55}>
            <div className="mt-16 flex flex-wrap items-center justify-center gap-x-8 gap-y-2">
              {[
                "Python",
                "PyTorch",
                "Qiskit",
                "sQUlearn",
                "scikit-learn",
                "TauREx 3",
              ].map((tech) => (
                <span
                  key={tech}
                  className={`${jetbrains.className} text-[10px] uppercase tracking-[0.2em] text-[#f5f0e8]/20`}
                >
                  {tech}
                </span>
              ))}
            </div>
          </Reveal>

          <Reveal delay={0.7}>
            <div className="mt-20">
              <p
                className={`${jetbrains.className} text-[10px] uppercase tracking-[0.3em] text-[#f5f0e8]/15`}
              >
                HACK-4-SAGES &middot; March 2026 &middot; Life Detection &amp;
                Biosignatures
              </p>
            </div>
          </Reveal>
        </div>

        {/* bottom fade */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-[#0a0a0a] to-transparent" />
      </section>
    </div>
  );
}
