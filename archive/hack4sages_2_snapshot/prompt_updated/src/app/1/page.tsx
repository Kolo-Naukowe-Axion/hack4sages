"use client";

import { useEffect, useRef, useState, ReactNode } from "react";
import { motion, useInView, AnimatePresence } from "framer-motion";
import { Cormorant_Garamond, Geist_Mono } from "next/font/google";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
  Tooltip,
} from "recharts";

const serif = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-serif",
});

const mono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

// ---------------------------------------------------------------------------
// Floating particles (CSS-driven, purely decorative)
// ---------------------------------------------------------------------------
function Particles() {
  const count = 60;
  const particles = useRef(
    Array.from({ length: count }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 2 + 0.5,
      duration: Math.random() * 80 + 40,
      delay: Math.random() * -80,
      opacity: Math.random() * 0.25 + 0.05,
    }))
  ).current;

  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
      {particles.map((p) => (
        <div
          key={p.id}
          className="absolute rounded-full"
          style={{
            left: `${p.x}%`,
            top: `${p.y}%`,
            width: p.size,
            height: p.size,
            background: `rgba(0, 212, 170, ${p.opacity})`,
            animation: `float-particle ${p.duration}s linear ${p.delay}s infinite`,
          }}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Grain overlay
// ---------------------------------------------------------------------------
function GrainOverlay() {
  return (
    <div
      className="pointer-events-none fixed inset-0 z-50 opacity-[0.03]"
      style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        backgroundRepeat: "repeat",
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Scroll-driven section reveal wrapper
// ---------------------------------------------------------------------------
function RevealSection({
  children,
  className = "",
  id,
}: {
  children: ReactNode;
  className?: string;
  id?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { amount: 0.3, once: false });

  return (
    <section
      ref={ref}
      id={id}
      className={`relative flex min-h-screen w-full snap-start items-center justify-center ${className}`}
    >
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
        transition={{ duration: 0.9, ease: [0.25, 0.1, 0.25, 1] }}
        className="w-full"
      >
        {children}
      </motion.div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Navigation dots
// ---------------------------------------------------------------------------
const SECTION_IDS = [
  "title",
  "problem",
  "architecture",
  "results",
  "molecules",
  "efficiency",
  "hardware",
  "summary",
];

const SECTION_LABELS = [
  "Title",
  "Problem",
  "Architecture",
  "Results",
  "Molecules",
  "Efficiency",
  "Hardware",
  "Summary",
];

function NavDots() {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const observers: IntersectionObserver[] = [];

    SECTION_IDS.forEach((id, idx) => {
      const el = document.getElementById(id);
      if (!el) return;

      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) setActive(idx);
        },
        { threshold: 0.4 }
      );
      observer.observe(el);
      observers.push(observer);
    });

    return () => observers.forEach((o) => o.disconnect());
  }, []);

  return (
    <nav className="fixed right-6 top-1/2 z-40 flex -translate-y-1/2 flex-col items-end gap-3">
      {SECTION_IDS.map((id, idx) => (
        <button
          key={id}
          onClick={() =>
            document.getElementById(id)?.scrollIntoView({ behavior: "smooth" })
          }
          className="group flex items-center gap-3"
          aria-label={`Go to ${SECTION_LABELS[idx]}`}
        >
          <span
            className={`text-xs font-mono tracking-wider uppercase transition-all duration-300 ${
              active === idx
                ? "text-[#00d4aa] opacity-100"
                : "text-white/30 opacity-0 group-hover:opacity-100"
            }`}
          >
            {SECTION_LABELS[idx]}
          </span>
          <span
            className={`block rounded-full transition-all duration-300 ${
              active === idx
                ? "h-3 w-3 bg-[#00d4aa] shadow-[0_0_12px_rgba(0,212,170,0.5)]"
                : "h-2 w-2 bg-white/20 group-hover:bg-white/40"
            }`}
          />
        </button>
      ))}
    </nav>
  );
}

// ---------------------------------------------------------------------------
// Animated counter
// ---------------------------------------------------------------------------
function AnimatedNumber({
  value,
  decimals = 3,
  duration = 2000,
  prefix = "",
}: {
  value: number;
  decimals?: number;
  duration?: number;
  prefix?: string;
}) {
  const [display, setDisplay] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { amount: 0.5, once: true });

  useEffect(() => {
    if (!inView) return;
    const start = performance.now();
    const animate = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(eased * value);
      if (progress < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [inView, value, duration]);

  return (
    <span ref={ref}>
      {prefix}
      {display.toFixed(decimals)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Glow box
// ---------------------------------------------------------------------------
function GlowBox({
  children,
  className = "",
  color = "0, 212, 170",
}: {
  children: ReactNode;
  className?: string;
  color?: string;
}) {
  return (
    <div
      className={`relative rounded-2xl border border-white/[0.06] bg-white/[0.02] backdrop-blur-sm ${className}`}
      style={{
        boxShadow: `0 0 80px -20px rgba(${color}, 0.08), inset 0 1px 0 rgba(255,255,255,0.03)`,
      }}
    >
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pipeline step component
// ---------------------------------------------------------------------------
function PipelineStep({
  label,
  detail,
  icon,
  delay,
}: {
  label: string;
  detail: string;
  icon: string;
  delay: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { amount: 0.5, once: true });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={inView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.6, delay, ease: [0.25, 0.1, 0.25, 1] }}
      className="flex flex-col items-center"
    >
      <div className="flex h-16 w-16 items-center justify-center rounded-xl border border-white/[0.08] bg-white/[0.03] text-2xl shadow-[0_0_40px_-10px_rgba(0,212,170,0.15)]">
        {icon}
      </div>
      <p className="mt-3 font-mono text-xs tracking-wider uppercase text-[#00d4aa]">
        {label}
      </p>
      <p className="mt-1 max-w-[140px] text-center text-xs text-white/40">
        {detail}
      </p>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Arrow connector
// ---------------------------------------------------------------------------
function Arrow() {
  return (
    <div className="flex items-center px-2 text-white/15">
      <svg width="40" height="12" viewBox="0 0 40 12" fill="none">
        <path
          d="M0 6H36M36 6L30 1M36 6L30 11"
          stroke="currentColor"
          strokeWidth="1.5"
        />
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Molecule bar data
// ---------------------------------------------------------------------------
const moleculeData = [
  { name: "H\u2082O", rmse: 0.218, fill: "#00d4aa" },
  { name: "CO\u2082", rmse: 0.261, fill: "#00b89a" },
  { name: "CH\u2084", rmse: 0.29, fill: "#009d83" },
  { name: "CO", rmse: 0.327, fill: "#00826d" },
  { name: "NH\u2083", rmse: 0.378, fill: "#006856" },
];

const comparisonData = [
  { name: "Random\nForest", mrmse: 1.2, color: "#333333" },
  { name: "Baseline\nCNN", mrmse: 0.85, color: "#444444" },
  { name: "ADC2023\nWinner", mrmse: 0.32, color: "#555555" },
  { name: "ExoBiome\nQELM", mrmse: 0.295, color: "#00d4aa" },
];

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------
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
    <div className="rounded-lg border border-white/10 bg-[#0a0a0a]/90 px-4 py-2 font-mono text-xs backdrop-blur">
      <p className="text-white/60">{label}</p>
      <p className="text-[#00d4aa]">{payload[0].value.toFixed(3)}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Spectrum visualization (decorative SVG)
// ---------------------------------------------------------------------------
function SpectrumViz() {
  const points = 52;
  const ref = useRef<SVGSVGElement>(null);
  const inView = useInView(ref, { amount: 0.5, once: true });

  const data = useRef(
    Array.from({ length: points }, (_, i) => {
      const x = i / (points - 1);
      return (
        0.5 +
        0.2 * Math.sin(x * 8) +
        0.15 * Math.cos(x * 13) +
        0.1 * Math.sin(x * 21) -
        0.08 * Math.cos(x * 5) +
        (Math.random() - 0.5) * 0.06
      );
    })
  ).current;

  const width = 600;
  const height = 120;
  const pathD = data
    .map((v, i) => {
      const x = (i / (points - 1)) * width;
      const y = height - v * height * 0.8 - height * 0.1;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <motion.svg
      ref={ref}
      viewBox={`0 0 ${width} ${height}`}
      className="w-full max-w-[600px] overflow-visible"
      initial={{ opacity: 0 }}
      animate={inView ? { opacity: 1 } : { opacity: 0 }}
      transition={{ duration: 1.2 }}
    >
      <defs>
        <linearGradient id="specGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#00d4aa" stopOpacity="0.6" />
          <stop offset="50%" stopColor="#00d4aa" stopOpacity="1" />
          <stop offset="100%" stopColor="#006856" stopOpacity="0.6" />
        </linearGradient>
        <linearGradient id="specFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#00d4aa" stopOpacity="0.15" />
          <stop offset="100%" stopColor="#00d4aa" stopOpacity="0" />
        </linearGradient>
      </defs>
      <motion.path
        d={`${pathD} L ${width} ${height} L 0 ${height} Z`}
        fill="url(#specFill)"
        initial={{ pathLength: 0 }}
        animate={inView ? { pathLength: 1 } : { pathLength: 0 }}
        transition={{ duration: 2, ease: "easeInOut" }}
      />
      <motion.path
        d={pathD}
        fill="none"
        stroke="url(#specGrad)"
        strokeWidth="2"
        initial={{ pathLength: 0 }}
        animate={inView ? { pathLength: 1 } : { pathLength: 0 }}
        transition={{ duration: 2, ease: "easeInOut" }}
      />
      {data.map((v, i) => {
        const x = (i / (points - 1)) * width;
        const y = height - v * height * 0.8 - height * 0.1;
        return (
          <motion.circle
            key={i}
            cx={x}
            cy={y}
            r="2"
            fill="#00d4aa"
            initial={{ opacity: 0 }}
            animate={inView ? { opacity: 0.5 } : { opacity: 0 }}
            transition={{ duration: 0.4, delay: 1.5 + i * 0.02 }}
          />
        );
      })}
      <text x="0" y={height + 14} fill="rgba(255,255,255,0.25)" fontSize="10" fontFamily="monospace">
        0.5 um
      </text>
      <text x={width - 40} y={height + 14} fill="rgba(255,255,255,0.25)" fontSize="10" fontFamily="monospace">
        7.8 um
      </text>
    </motion.svg>
  );
}

// ---------------------------------------------------------------------------
// Quantum circuit decoration (SVG)
// ---------------------------------------------------------------------------
function QuantumCircuitViz() {
  const ref = useRef<SVGSVGElement>(null);
  const inView = useInView(ref, { amount: 0.5, once: true });
  const qubits = 6;
  const width = 400;
  const height = 180;
  const gap = height / (qubits + 1);

  return (
    <motion.svg
      ref={ref}
      viewBox={`0 0 ${width} ${height}`}
      className="w-full max-w-[400px] overflow-visible opacity-60"
      initial={{ opacity: 0 }}
      animate={inView ? { opacity: 0.6 } : { opacity: 0 }}
      transition={{ duration: 1 }}
    >
      {Array.from({ length: qubits }, (_, i) => {
        const y = gap * (i + 1);
        return (
          <g key={i}>
            <motion.line
              x1="20"
              y1={y}
              x2={width - 20}
              y2={y}
              stroke="rgba(0,212,170,0.2)"
              strokeWidth="1"
              initial={{ pathLength: 0 }}
              animate={inView ? { pathLength: 1 } : { pathLength: 0 }}
              transition={{ duration: 1.5, delay: i * 0.1 }}
            />
            <text
              x="5"
              y={y + 4}
              fill="rgba(0,212,170,0.3)"
              fontSize="9"
              fontFamily="monospace"
            >
              q{i}
            </text>
            {/* RY gates */}
            {[100, 220, 340].map((gx, gi) => (
              <motion.rect
                key={gi}
                x={gx - 14}
                y={y - 10}
                width="28"
                height="20"
                rx="3"
                fill="none"
                stroke="rgba(0,212,170,0.25)"
                strokeWidth="1"
                initial={{ opacity: 0 }}
                animate={inView ? { opacity: 1 } : { opacity: 0 }}
                transition={{ duration: 0.4, delay: 0.8 + gi * 0.2 + i * 0.05 }}
              />
            ))}
            <motion.text
              x="100"
              y={y + 3}
              textAnchor="middle"
              fill="rgba(0,212,170,0.4)"
              fontSize="8"
              fontFamily="monospace"
              initial={{ opacity: 0 }}
              animate={inView ? { opacity: 1 } : { opacity: 0 }}
              transition={{ duration: 0.4, delay: 1 }}
            >
              RY
            </motion.text>
            <motion.text
              x="220"
              y={y + 3}
              textAnchor="middle"
              fill="rgba(0,212,170,0.4)"
              fontSize="8"
              fontFamily="monospace"
              initial={{ opacity: 0 }}
              animate={inView ? { opacity: 1 } : { opacity: 0 }}
              transition={{ duration: 0.4, delay: 1.2 }}
            >
              RZ
            </motion.text>
            <motion.text
              x="340"
              y={y + 3}
              textAnchor="middle"
              fill="rgba(0,212,170,0.4)"
              fontSize="8"
              fontFamily="monospace"
              initial={{ opacity: 0 }}
              animate={inView ? { opacity: 1 } : { opacity: 0 }}
              transition={{ duration: 0.4, delay: 1.4 }}
            >
              RY
            </motion.text>
            {/* CNOT connections */}
            {i < qubits - 1 && (
              <motion.line
                x1="160"
                y1={y}
                x2="160"
                y2={y + gap}
                stroke="rgba(0,212,170,0.15)"
                strokeWidth="1"
                strokeDasharray="3 3"
                initial={{ pathLength: 0 }}
                animate={inView ? { pathLength: 1 } : { pathLength: 0 }}
                transition={{ duration: 0.6, delay: 1.5 + i * 0.1 }}
              />
            )}
          </g>
        );
      })}
    </motion.svg>
  );
}

// ---------------------------------------------------------------------------
// Page header fade in
// ---------------------------------------------------------------------------
function FadeInText({
  children,
  delay = 0,
  className = "",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 1, delay, ease: [0.25, 0.1, 0.25, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Stagger reveal for list items
// ---------------------------------------------------------------------------
function StaggerItem({
  children,
  index,
  className = "",
}: {
  children: ReactNode;
  index: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { amount: 0.3, once: true });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 24 }}
      transition={{
        duration: 0.6,
        delay: index * 0.12,
        ease: [0.25, 0.1, 0.25, 1],
      }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// ===========================================================================
// MAIN PAGE
// ===========================================================================
export default function ExoBiomePage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div
      className={`${serif.variable} ${mono.variable} relative min-h-screen snap-y snap-mandatory overflow-y-auto scroll-smooth`}
      style={{ background: "#050505" }}
    >
      <style jsx global>{`
        @keyframes float-particle {
          0% {
            transform: translate(0, 0);
          }
          25% {
            transform: translate(30px, -40px);
          }
          50% {
            transform: translate(-20px, -80px);
          }
          75% {
            transform: translate(40px, -40px);
          }
          100% {
            transform: translate(0, 0);
          }
        }

        @keyframes pulse-glow {
          0%, 100% {
            opacity: 0.4;
          }
          50% {
            opacity: 1;
          }
        }

        .font-serif {
          font-family: var(--font-serif), Georgia, serif;
        }
        .font-mono {
          font-family: var(--font-mono), ui-monospace, monospace;
        }

        ::-webkit-scrollbar {
          width: 0px;
        }
      `}</style>

      <GrainOverlay />
      {mounted && <Particles />}
      <NavDots />

      {/* ================================================================= */}
      {/* TITLE                                                             */}
      {/* ================================================================= */}
      <section
        id="title"
        className="relative flex min-h-screen snap-start flex-col items-center justify-center px-6"
      >
        <div className="absolute inset-0 flex items-center justify-center">
          <div
            className="h-[500px] w-[500px] rounded-full"
            style={{
              background:
                "radial-gradient(circle, rgba(0,212,170,0.06) 0%, transparent 70%)",
            }}
          />
        </div>

        <FadeInText delay={0.2}>
          <p className="font-mono text-xs tracking-[0.3em] uppercase text-[#00d4aa]/60">
            HACK-4-SAGES 2026 &middot; ETH Zurich
          </p>
        </FadeInText>

        <FadeInText delay={0.5}>
          <h1 className="font-serif mt-6 text-center text-7xl font-light leading-[0.95] tracking-tight text-white md:text-8xl lg:text-9xl">
            Exo
            <span className="text-[#00d4aa]">Biome</span>
          </h1>
        </FadeInText>

        <FadeInText delay={0.9}>
          <p className="font-serif mt-8 max-w-xl text-center text-xl font-light leading-relaxed text-white/40 md:text-2xl">
            Quantum-enhanced biosignature detection
            <br />
            from exoplanet transmission spectra
          </p>
        </FadeInText>

        <FadeInText delay={1.3}>
          <div className="mt-12 flex items-center gap-8 font-mono text-[10px] tracking-widest uppercase text-white/20">
            <span>12 Qubits</span>
            <span className="h-px w-8 bg-white/10" />
            <span>5 Molecules</span>
            <span className="h-px w-8 bg-white/10" />
            <span>0.295 mRMSE</span>
          </div>
        </FadeInText>

        <FadeInText delay={1.7}>
          <motion.div
            className="mt-20 text-white/15"
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path
                d="M4 7L10 13L16 7"
                stroke="currentColor"
                strokeWidth="1.5"
              />
            </svg>
          </motion.div>
        </FadeInText>
      </section>

      {/* ================================================================= */}
      {/* PROBLEM                                                           */}
      {/* ================================================================= */}
      <RevealSection id="problem">
        <div className="mx-auto max-w-4xl px-6">
          <p className="font-mono text-xs tracking-[0.25em] uppercase text-[#00d4aa]/50">
            The Challenge
          </p>
          <h2 className="font-serif mt-4 text-4xl font-light leading-tight text-white/90 md:text-5xl lg:text-6xl">
            Reading the atmosphere
            <br />
            <span className="text-white/40">of worlds we cannot visit</span>
          </h2>
          <div className="mt-12 grid gap-12 md:grid-cols-2">
            <div>
              <p className="font-serif text-lg font-light leading-relaxed text-white/35">
                When starlight filters through an exoplanet&apos;s atmosphere,
                each molecule leaves a unique fingerprint in the transmission
                spectrum. Decoding these faint signals from noisy data is one of
                the hardest inverse problems in astrophysics.
              </p>
              <p className="font-serif mt-6 text-lg font-light leading-relaxed text-white/35">
                ExoBiome uses a hybrid quantum-classical neural network to
                predict molecular concentrations directly from spectral
                observations&mdash;achieving state-of-the-art accuracy with
                radical efficiency.
              </p>
            </div>
            <div className="flex flex-col items-center justify-center">
              <SpectrumViz />
              <p className="font-mono mt-4 text-[10px] tracking-widest uppercase text-white/20">
                52 wavelength bins &middot; 0.5&ndash;7.8 &mu;m
              </p>
            </div>
          </div>
        </div>
      </RevealSection>

      {/* ================================================================= */}
      {/* ARCHITECTURE                                                      */}
      {/* ================================================================= */}
      <RevealSection id="architecture">
        <div className="mx-auto max-w-5xl px-6">
          <p className="font-mono text-center text-xs tracking-[0.25em] uppercase text-[#00d4aa]/50">
            Architecture
          </p>
          <h2 className="font-serif mt-4 text-center text-4xl font-light text-white/90 md:text-5xl">
            End-to-end pipeline
          </h2>

          {/* Pipeline diagram */}
          <div className="mt-16 flex flex-wrap items-start justify-center gap-y-8">
            <PipelineStep
              icon="~"
              label="Spectrum"
              detail="52-bin transmission spectrum input"
              delay={0}
            />
            <Arrow />
            <PipelineStep
              icon="&#x25A4;"
              label="SpectralEncoder"
              detail="Conv1D feature extraction"
              delay={0.1}
            />
            <Arrow />
            <PipelineStep
              icon="&#x2295;"
              label="Fusion"
              detail="Spectral + auxiliary merge"
              delay={0.2}
            />
            <Arrow />
            <PipelineStep
              icon="&#x29BE;"
              label="Quantum Circuit"
              detail="12 qubits, 2 layers, RY/CNOT/RZ"
              delay={0.3}
            />
            <Arrow />
            <PipelineStep
              icon="&#x2261;"
              label="Prediction"
              detail="log&#8321;&#8320; VMR for 5 molecules"
              delay={0.4}
            />
          </div>

          {/* Quantum circuit visualization */}
          <div className="mt-16 flex flex-col items-center">
            <QuantumCircuitViz />
            <p className="font-mono mt-4 text-[10px] tracking-widest text-white/20">
              VARIATIONAL QUANTUM CIRCUIT &middot; RY + CNOT + RZ ENTANGLING
              LAYERS
            </p>
          </div>

          {/* Architecture details */}
          <div className="mt-16 grid gap-6 md:grid-cols-3">
            {[
              {
                label: "SpectralEncoder",
                value: "Conv1D",
                desc: "3-layer convolutional network extracts spectral features from raw 52-bin input",
              },
              {
                label: "AuxEncoder",
                value: "FFN",
                desc: "Feed-forward network processes stellar parameters, planet radius, orbital data",
              },
              {
                label: "Quantum Layer",
                value: "QELM",
                desc: "Quantum Extreme Learning Machine with 12 qubits and 2 variational layers",
              },
            ].map((item, i) => (
              <StaggerItem key={item.label} index={i}>
                <GlowBox className="p-6">
                  <p className="font-mono text-[10px] tracking-widest uppercase text-[#00d4aa]/60">
                    {item.label}
                  </p>
                  <p className="font-serif mt-2 text-2xl font-light text-white/80">
                    {item.value}
                  </p>
                  <p className="mt-2 text-xs leading-relaxed text-white/30">
                    {item.desc}
                  </p>
                </GlowBox>
              </StaggerItem>
            ))}
          </div>
        </div>
      </RevealSection>

      {/* ================================================================= */}
      {/* RESULTS                                                           */}
      {/* ================================================================= */}
      <RevealSection id="results">
        <div className="mx-auto max-w-5xl px-6">
          <p className="font-mono text-center text-xs tracking-[0.25em] uppercase text-[#00d4aa]/50">
            Performance
          </p>
          <h2 className="font-serif mt-4 text-center text-4xl font-light text-white/90 md:text-5xl">
            Benchmark results
          </h2>

          <div className="mt-16 flex flex-col items-center">
            {/* Big hero number */}
            <div className="text-center">
              <p className="font-mono text-xs tracking-widest uppercase text-white/30">
                Mean Relative RMSE
              </p>
              <p className="font-serif mt-2 text-8xl font-light tracking-tight text-[#00d4aa] md:text-9xl">
                <AnimatedNumber value={0.295} />
              </p>
              <p className="font-mono mt-2 text-xs text-white/20">
                mRMSE across 5 target molecules
              </p>
            </div>

            {/* Comparison chart */}
            <div className="mt-16 w-full max-w-2xl">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart
                  data={comparisonData}
                  margin={{ top: 20, right: 20, bottom: 40, left: 20 }}
                  barCategoryGap="25%"
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#111"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 11, fontFamily: "monospace" }}
                    axisLine={{ stroke: "#222" }}
                    tickLine={false}
                    interval={0}
                  />
                  <YAxis
                    tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 10, fontFamily: "monospace" }}
                    axisLine={{ stroke: "#222" }}
                    tickLine={false}
                    domain={[0, 1.4]}
                  />
                  <Tooltip
                    content={<CustomTooltip />}
                    cursor={{ fill: "rgba(255,255,255,0.02)" }}
                  />
                  <Bar dataKey="mrmse" radius={[4, 4, 0, 0]} maxBarSize={60}>
                    {comparisonData.map((entry, index) => (
                      <Cell key={index} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="font-mono mt-2 text-center text-[10px] tracking-widest uppercase text-white/20">
                Ariel Data Challenge 2023 Benchmark &middot; Lower is better
              </p>
            </div>
          </div>
        </div>
      </RevealSection>

      {/* ================================================================= */}
      {/* PER-MOLECULE BREAKDOWN                                            */}
      {/* ================================================================= */}
      <RevealSection id="molecules">
        <div className="mx-auto max-w-5xl px-6">
          <p className="font-mono text-center text-xs tracking-[0.25em] uppercase text-[#00d4aa]/50">
            Per-Molecule Accuracy
          </p>
          <h2 className="font-serif mt-4 text-center text-4xl font-light text-white/90 md:text-5xl">
            Five molecules, one model
          </h2>

          <div className="mt-16 grid gap-4 md:grid-cols-5">
            {moleculeData.map((mol, i) => (
              <StaggerItem key={mol.name} index={i}>
                <GlowBox className="flex flex-col items-center p-6 text-center">
                  <p className="font-mono text-xs tracking-widest uppercase text-white/40">
                    {mol.name}
                  </p>
                  <p
                    className="font-serif mt-3 text-4xl font-light"
                    style={{ color: mol.fill }}
                  >
                    <AnimatedNumber value={mol.rmse} />
                  </p>
                  <p className="font-mono mt-1 text-[10px] text-white/20">
                    RMSE
                  </p>
                  {/* Mini bar */}
                  <div className="mt-4 h-1 w-full overflow-hidden rounded-full bg-white/[0.05]">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ background: mol.fill }}
                      initial={{ width: 0 }}
                      whileInView={{ width: `${(mol.rmse / 0.5) * 100}%` }}
                      transition={{ duration: 1.2, delay: 0.3 + i * 0.1 }}
                      viewport={{ once: true }}
                    />
                  </div>
                </GlowBox>
              </StaggerItem>
            ))}
          </div>

          {/* Molecule chart */}
          <div className="mt-12 w-full max-w-2xl mx-auto">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={moleculeData}
                margin={{ top: 10, right: 20, bottom: 20, left: 20 }}
                barCategoryGap="20%"
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#111"
                  vertical={false}
                />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 12, fontFamily: "monospace" }}
                  axisLine={{ stroke: "#222" }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 10, fontFamily: "monospace" }}
                  axisLine={{ stroke: "#222" }}
                  tickLine={false}
                  domain={[0, 0.5]}
                />
                <Tooltip
                  content={<CustomTooltip />}
                  cursor={{ fill: "rgba(255,255,255,0.02)" }}
                />
                <Bar dataKey="rmse" radius={[4, 4, 0, 0]} maxBarSize={50}>
                  {moleculeData.map((entry, index) => (
                    <Cell key={index} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <p className="font-mono mt-4 text-center text-[10px] tracking-widest uppercase text-white/20">
            log&#8321;&#8320; VMR prediction accuracy &middot; 41,423 spectra
          </p>
        </div>
      </RevealSection>

      {/* ================================================================= */}
      {/* TRAINING EFFICIENCY                                               */}
      {/* ================================================================= */}
      <RevealSection id="efficiency">
        <div className="mx-auto max-w-4xl px-6">
          <p className="font-mono text-center text-xs tracking-[0.25em] uppercase text-[#00d4aa]/50">
            Efficiency
          </p>
          <h2 className="font-serif mt-4 text-center text-4xl font-light text-white/90 md:text-5xl">
            Radical simplicity
          </h2>
          <p className="font-serif mx-auto mt-6 max-w-xl text-center text-lg font-light leading-relaxed text-white/30">
            State-of-the-art accuracy achieved with a fraction of the
            computational resources typically required for atmospheric retrieval.
          </p>

          <div className="mt-16 grid gap-6 md:grid-cols-3">
            {[
              {
                value: "~3",
                unit: "min",
                label: "Training Time",
                desc: "Full training cycle on standard GPU hardware",
              },
              {
                value: "120",
                unit: "K",
                label: "Parameters",
                desc: "Orders of magnitude fewer than deep retrieval models",
              },
              {
                value: "12",
                unit: "qubits",
                label: "Quantum Circuit",
                desc: "Compact variational circuit, NISQ-compatible",
              },
            ].map((stat, i) => (
              <StaggerItem key={stat.label} index={i}>
                <GlowBox className="p-8 text-center">
                  <p className="font-serif text-6xl font-light text-white/90">
                    {stat.value}
                    <span className="font-mono text-xl text-[#00d4aa]/70">
                      {stat.unit}
                    </span>
                  </p>
                  <p className="font-mono mt-3 text-xs tracking-widest uppercase text-[#00d4aa]/50">
                    {stat.label}
                  </p>
                  <p className="mt-2 text-xs leading-relaxed text-white/25">
                    {stat.desc}
                  </p>
                </GlowBox>
              </StaggerItem>
            ))}
          </div>

          <div className="mt-16 grid gap-6 md:grid-cols-2">
            <StaggerItem index={0}>
              <GlowBox className="p-6">
                <p className="font-mono text-[10px] tracking-widest uppercase text-[#00d4aa]/50">
                  Dataset
                </p>
                <p className="font-serif mt-2 text-2xl font-light text-white/80">
                  Ariel Data Challenge 2023
                </p>
                <div className="mt-4 space-y-2 font-mono text-xs text-white/30">
                  <div className="flex justify-between">
                    <span>Spectra</span>
                    <span className="text-white/50">41,423</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Wavelength bins</span>
                    <span className="text-white/50">52</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Range</span>
                    <span className="text-white/50">
                      0.5 &ndash; 7.8 &mu;m
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Targets</span>
                    <span className="text-white/50">
                      H&#8322;O, CO&#8322;, CO, CH&#8324;, NH&#8323;
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Format</span>
                    <span className="text-white/50">log&#8321;&#8320; VMR</span>
                  </div>
                </div>
              </GlowBox>
            </StaggerItem>
            <StaggerItem index={1}>
              <GlowBox className="p-6">
                <p className="font-mono text-[10px] tracking-widest uppercase text-[#00d4aa]/50">
                  Why Quantum?
                </p>
                <p className="font-serif mt-2 text-2xl font-light text-white/80">
                  Quantum advantage
                </p>
                <div className="mt-4 space-y-3 text-xs leading-relaxed text-white/30">
                  <p>
                    The Quantum Extreme Learning Machine (QELM) leverages the
                    exponentially large Hilbert space of a 12-qubit system to
                    create a rich, high-dimensional feature map from compressed
                    spectral data.
                  </p>
                  <p>
                    This enables the model to capture complex nonlinear
                    relationships between spectral features and molecular
                    concentrations&mdash;with dramatically fewer trainable
                    parameters than classical alternatives.
                  </p>
                  <p>
                    First-ever application of quantum machine learning to
                    biosignature detection in exoplanet atmospheres.
                  </p>
                </div>
              </GlowBox>
            </StaggerItem>
          </div>
        </div>
      </RevealSection>

      {/* ================================================================= */}
      {/* HARDWARE                                                          */}
      {/* ================================================================= */}
      <RevealSection id="hardware">
        <div className="mx-auto max-w-4xl px-6">
          <p className="font-mono text-center text-xs tracking-[0.25em] uppercase text-[#00d4aa]/50">
            Hardware
          </p>
          <h2 className="font-serif mt-4 text-center text-4xl font-light text-white/90 md:text-5xl">
            Real quantum hardware
          </h2>
          <p className="font-serif mx-auto mt-6 max-w-xl text-center text-lg font-light leading-relaxed text-white/30">
            Validated on production quantum processors&mdash;not just
            simulators.
          </p>

          <div className="mt-16 grid gap-6 md:grid-cols-2">
            {[
              {
                name: "IQM Spark",
                location: "Odra 5 &middot; PWR Wroc\u0142aw",
                qubits: "5",
                detail:
                  "Local superconducting quantum processor, Qiskit-on-IQM SDK integration",
              },
              {
                name: "VTT Q50",
                location: "VTT Finland &middot; Remote access",
                qubits: "53",
                detail:
                  "High-qubit-count processor for scaling experiments via PWR partnership",
              },
            ].map((hw, i) => (
              <StaggerItem key={hw.name} index={i}>
                <GlowBox className="p-8">
                  <div className="flex items-baseline justify-between">
                    <p className="font-serif text-3xl font-light text-white/85">
                      {hw.name}
                    </p>
                    <p className="font-mono text-2xl text-[#00d4aa]/70">
                      {hw.qubits}
                      <span className="text-xs text-[#00d4aa]/40"> qubits</span>
                    </p>
                  </div>
                  <p
                    className="font-mono mt-2 text-xs tracking-wider text-white/30"
                    dangerouslySetInnerHTML={{ __html: hw.location }}
                  />
                  <p className="mt-4 text-sm leading-relaxed text-white/25">
                    {hw.detail}
                  </p>
                </GlowBox>
              </StaggerItem>
            ))}
          </div>

          <div className="mt-12">
            <StaggerItem index={0}>
              <GlowBox className="p-6">
                <p className="font-mono text-[10px] tracking-widest uppercase text-[#00d4aa]/50">
                  Tech Stack
                </p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {[
                    "Python",
                    "PyTorch",
                    "Qiskit",
                    "qiskit-on-iqm",
                    "sQUlearn",
                    "scikit-learn",
                    "NumPy",
                    "SciPy",
                    "Matplotlib",
                    "Jupyter",
                  ].map((tech) => (
                    <span
                      key={tech}
                      className="rounded-full border border-white/[0.06] bg-white/[0.02] px-3 py-1 font-mono text-[11px] text-white/35"
                    >
                      {tech}
                    </span>
                  ))}
                </div>
              </GlowBox>
            </StaggerItem>
          </div>
        </div>
      </RevealSection>

      {/* ================================================================= */}
      {/* SUMMARY                                                           */}
      {/* ================================================================= */}
      <RevealSection id="summary">
        <div className="mx-auto max-w-4xl px-6 text-center">
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div
              className="h-[600px] w-[600px] rounded-full"
              style={{
                background:
                  "radial-gradient(circle, rgba(0,212,170,0.04) 0%, transparent 70%)",
              }}
            />
          </div>

          <p className="font-mono text-xs tracking-[0.25em] uppercase text-[#00d4aa]/50">
            Summary
          </p>
          <h2 className="font-serif mt-6 text-5xl font-light leading-tight text-white/90 md:text-6xl lg:text-7xl">
            The first quantum model
            <br />
            <span className="text-[#00d4aa]">for biosignature detection</span>
          </h2>

          <div className="mx-auto mt-12 max-w-2xl">
            <div className="grid grid-cols-3 gap-px overflow-hidden rounded-xl border border-white/[0.06]">
              {[
                { value: "0.295", label: "mRMSE" },
                { value: "120K", label: "Parameters" },
                { value: "~3 min", label: "Training" },
              ].map((stat) => (
                <div key={stat.label} className="bg-white/[0.02] p-6">
                  <p className="font-serif text-3xl font-light text-[#00d4aa]">
                    {stat.value}
                  </p>
                  <p className="font-mono mt-1 text-[10px] tracking-widest uppercase text-white/30">
                    {stat.label}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-12 space-y-3">
            {[
              "Outperforms the ADC2023 competition winner",
              "First application of quantum ML to exoplanet biosignatures",
              "Validated on real quantum hardware (IQM Spark, VTT Q50)",
              "NISQ-compatible: runs on current-generation quantum processors",
            ].map((point, i) => (
              <StaggerItem key={i} index={i} className="flex justify-center">
                <p className="font-serif text-lg font-light text-white/35">
                  <span className="mr-3 text-[#00d4aa]/40">&mdash;</span>
                  {point}
                </p>
              </StaggerItem>
            ))}
          </div>

          <motion.div
            className="mt-20"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.5 }}
            viewport={{ once: true }}
          >
            <p className="font-mono text-xs tracking-[0.2em] uppercase text-white/15">
              HACK-4-SAGES 2026 &middot; Life Detection and Biosignatures
            </p>
            <p className="font-mono mt-2 text-[10px] tracking-widest text-white/10">
              ETH Zurich &middot; Origins Federation
            </p>
          </motion.div>
        </div>
      </RevealSection>

      {/* Bottom spacer */}
      <div className="h-20" />
    </div>
  );
}
