"use client";

import { useRef, useEffect, useState, useMemo } from "react";
import { DM_Sans, JetBrains_Mono } from "next/font/google";
import { motion, useScroll, useTransform, useSpring, useInView, AnimatePresence } from "framer-motion";
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

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
  weight: ["300", "400", "500", "600", "700"],
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  weight: ["300", "400", "500", "700"],
});

// ---------- SPECTRUM DATA ----------
// Realistic transmission spectrum: 52 wavelength bins from 0.5 to 7.8 μm
// Values represent transit depth (normalized 0–1 scale)

const WAVELENGTHS = Array.from({ length: 52 }, (_, i) => 0.5 + i * (7.3 / 51));

function generateSpectrum(): number[] {
  const baseline = 0.52;
  return WAVELENGTHS.map((wl) => {
    let depth = baseline;
    // H₂O features (~0.94, ~1.15, ~1.4, ~1.9, ~2.7, ~6.3 μm)
    depth += 0.06 * Math.exp(-((wl - 0.94) ** 2) / 0.005);
    depth += 0.08 * Math.exp(-((wl - 1.15) ** 2) / 0.008);
    depth += 0.12 * Math.exp(-((wl - 1.4) ** 2) / 0.012);
    depth += 0.15 * Math.exp(-((wl - 1.9) ** 2) / 0.02);
    depth += 0.18 * Math.exp(-((wl - 2.7) ** 2) / 0.04);
    depth += 0.10 * Math.exp(-((wl - 6.3) ** 2) / 0.06);
    // CO₂ features (~2.0, ~4.3 μm)
    depth += 0.10 * Math.exp(-((wl - 2.0) ** 2) / 0.015);
    depth += 0.22 * Math.exp(-((wl - 4.3) ** 2) / 0.03);
    // CO feature (~4.7 μm)
    depth += 0.09 * Math.exp(-((wl - 4.7) ** 2) / 0.02);
    // CH₄ features (~2.3, ~3.3, ~7.7 μm)
    depth += 0.11 * Math.exp(-((wl - 2.3) ** 2) / 0.015);
    depth += 0.16 * Math.exp(-((wl - 3.3) ** 2) / 0.025);
    depth += 0.08 * Math.exp(-((wl - 7.7) ** 2) / 0.04);
    // NH₃ features (~2.0, ~6.1, ~10.5 μm)
    depth += 0.07 * Math.exp(-((wl - 2.0) ** 2) / 0.01);
    depth += 0.09 * Math.exp(-((wl - 6.1) ** 2) / 0.04);
    return depth;
  });
}

const SPECTRUM = generateSpectrum();
const SPECTRUM_MIN = Math.min(...SPECTRUM);
const SPECTRUM_MAX = Math.max(...SPECTRUM);

// Molecule absorption labels with their primary wavelength positions
const MOLECULE_LABELS: { name: string; wl: number; color: string }[] = [
  { name: "H₂O", wl: 1.4, color: "#3b82f6" },
  { name: "H₂O", wl: 2.7, color: "#3b82f6" },
  { name: "CO₂", wl: 4.3, color: "#22d3ee" },
  { name: "CO", wl: 4.7, color: "#10b981" },
  { name: "CH₄", wl: 3.3, color: "#eab308" },
  { name: "NH₃", wl: 6.1, color: "#f97316" },
  { name: "H₂O", wl: 6.3, color: "#3b82f6" },
];

// Wavelength to color mapping (visible + IR representation)
function wavelengthToColor(wl: number): string {
  const t = (wl - 0.5) / 7.3;
  if (t < 0.15) return "#4338ca"; // deep violet-blue
  if (t < 0.3) return "#3b82f6"; // blue
  if (t < 0.45) return "#22d3ee"; // cyan
  if (t < 0.6) return "#10b981"; // green
  if (t < 0.75) return "#eab308"; // yellow
  if (t < 0.9) return "#f97316"; // orange
  return "#ef4444"; // red
}

// ---------- SVG SPECTRUM COMPONENT ----------
function SpectrumSVG({
  breathePhase,
  opacity = 1,
  glowIntensity = 1,
  showLabels = true,
  highlightMolecule,
  className = "",
}: {
  breathePhase: number;
  opacity?: number;
  glowIntensity?: number;
  showLabels?: boolean;
  highlightMolecule?: string;
  className?: string;
}) {
  const svgW = 1200;
  const svgH = 300;
  const padX = 60;
  const padY = 30;
  const plotW = svgW - padX * 2;
  const plotH = svgH - padY * 2;

  const points = useMemo(() => {
    return WAVELENGTHS.map((wl, i) => {
      const x = padX + (i / (WAVELENGTHS.length - 1)) * plotW;
      const breathOffset = Math.sin(breathePhase + wl * 0.8) * 3;
      const normVal = (SPECTRUM[i] - SPECTRUM_MIN) / (SPECTRUM_MAX - SPECTRUM_MIN);
      const y = padY + (1 - normVal) * plotH + breathOffset;
      return { x, y, wl };
    });
  }, [breathePhase, plotW, plotH]);

  const pathD = useMemo(() => {
    if (points.length < 2) return "";
    let d = `M ${points[0].x},${points[0].y}`;
    for (let i = 1; i < points.length; i++) {
      const prev = points[i - 1];
      const curr = points[i];
      const cpx1 = prev.x + (curr.x - prev.x) * 0.4;
      const cpx2 = prev.x + (curr.x - prev.x) * 0.6;
      d += ` C ${cpx1},${prev.y} ${cpx2},${curr.y} ${curr.x},${curr.y}`;
    }
    return d;
  }, [points]);

  const areaD = useMemo(() => {
    if (points.length < 2) return "";
    return (
      pathD +
      ` L ${points[points.length - 1].x},${padY + plotH} L ${points[0].x},${padY + plotH} Z`
    );
  }, [pathD, points, plotH]);

  const gradId = "specGrad-" + (highlightMolecule || "main");
  const glowId = "specGlow-" + (highlightMolecule || "main");
  const areaGradId = "areaGrad-" + (highlightMolecule || "main");

  return (
    <svg
      viewBox={`0 0 ${svgW} ${svgH}`}
      className={className}
      style={{ opacity }}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#4338ca" />
          <stop offset="15%" stopColor="#3b82f6" />
          <stop offset="30%" stopColor="#22d3ee" />
          <stop offset="50%" stopColor="#10b981" />
          <stop offset="70%" stopColor="#eab308" />
          <stop offset="85%" stopColor="#f97316" />
          <stop offset="100%" stopColor="#ef4444" />
        </linearGradient>
        <linearGradient id={areaGradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.15 * glowIntensity} />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
        </linearGradient>
        <filter id={glowId}>
          <feGaussianBlur stdDeviation={3 * glowIntensity} result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Area fill under curve */}
      <path d={areaD} fill={`url(#${areaGradId})`} />

      {/* Main spectrum curve */}
      <path
        d={pathD}
        fill="none"
        stroke={`url(#${gradId})`}
        strokeWidth={2.5}
        filter={`url(#${glowId})`}
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Molecule labels */}
      {showLabels &&
        MOLECULE_LABELS.map((mol, idx) => {
          const t = (mol.wl - 0.5) / 7.3;
          const x = padX + t * plotW;
          const isHighlighted =
            !highlightMolecule || highlightMolecule === mol.name;
          const labelOpacity = isHighlighted ? 1 : 0.15;

          return (
            <g key={`${mol.name}-${idx}`} opacity={labelOpacity}>
              <line
                x1={x}
                y1={padY}
                x2={x}
                y2={padY + plotH}
                stroke={mol.color}
                strokeWidth={1}
                strokeDasharray="3 4"
                opacity={0.5}
              />
              <text
                x={x}
                y={padY - 8}
                textAnchor="middle"
                fill={mol.color}
                fontSize={11}
                fontFamily="var(--font-jetbrains), monospace"
                fontWeight={500}
              >
                {mol.name}
              </text>
              <text
                x={x}
                y={padY + plotH + 16}
                textAnchor="middle"
                fill="#6b7280"
                fontSize={9}
                fontFamily="var(--font-jetbrains), monospace"
              >
                {mol.wl.toFixed(1)}μm
              </text>
            </g>
          );
        })}

      {/* Axis labels */}
      <text
        x={svgW / 2}
        y={svgH - 2}
        textAnchor="middle"
        fill="#4b5563"
        fontSize={10}
        fontFamily="var(--font-jetbrains), monospace"
      >
        Wavelength (μm)
      </text>
      <text
        x={12}
        y={svgH / 2}
        textAnchor="middle"
        fill="#4b5563"
        fontSize={10}
        fontFamily="var(--font-jetbrains), monospace"
        transform={`rotate(-90, 12, ${svgH / 2})`}
      >
        Transit Depth
      </text>
    </svg>
  );
}

// ---------- MINI SPECTRUM FOR MOLECULE ----------
function MoleculeSpectrum({
  molecule,
  color,
  primaryWl,
}: {
  molecule: string;
  color: string;
  primaryWl: number;
}) {
  const svgW = 300;
  const svgH = 80;
  const padX = 10;
  const padY = 8;
  const plotW = svgW - padX * 2;
  const plotH = svgH - padY * 2;

  const points = WAVELENGTHS.map((wl, i) => {
    const x = padX + (i / (WAVELENGTHS.length - 1)) * plotW;
    const normVal = (SPECTRUM[i] - SPECTRUM_MIN) / (SPECTRUM_MAX - SPECTRUM_MIN);
    const y = padY + (1 - normVal) * plotH;
    return { x, y, wl };
  });

  let pathD = `M ${points[0].x},${points[0].y}`;
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    const cpx1 = prev.x + (curr.x - prev.x) * 0.4;
    const cpx2 = prev.x + (curr.x - prev.x) * 0.6;
    pathD += ` C ${cpx1},${prev.y} ${cpx2},${curr.y} ${curr.x},${curr.y}`;
  }

  const hlX = padX + ((primaryWl - 0.5) / 7.3) * plotW;

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full h-auto">
      <defs>
        <linearGradient id={`miniGrad-${molecule}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.3} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
        <radialGradient id={`hlGlow-${molecule}`} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={color} stopOpacity={0.4} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </radialGradient>
      </defs>

      {/* Highlight region */}
      <rect
        x={hlX - 20}
        y={padY}
        width={40}
        height={plotH}
        fill={`url(#hlGlow-${molecule})`}
      />

      {/* Spectrum line */}
      <path
        d={pathD}
        fill="none"
        stroke="#374151"
        strokeWidth={1.5}
        opacity={0.5}
      />

      {/* Highlighted feature line */}
      <line
        x1={hlX}
        y1={padY}
        x2={hlX}
        y2={padY + plotH}
        stroke={color}
        strokeWidth={1.5}
        strokeDasharray="2 3"
      />
    </svg>
  );
}

// ---------- ANIMATED COUNTER ----------
function AnimatedNumber({
  value,
  decimals = 3,
  duration = 2,
}: {
  value: number;
  decimals?: number;
  duration?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const [display, setDisplay] = useState("0.000");

  useEffect(() => {
    if (!inView) return;
    const start = Date.now();
    const durationMs = duration * 1000;
    const animate = () => {
      const elapsed = Date.now() - start;
      const progress = Math.min(elapsed / durationMs, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay((value * eased).toFixed(decimals));
      if (progress < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [inView, value, decimals, duration]);

  return <span ref={ref}>{display}</span>;
}

// ---------- SECTION WRAPPER ----------
function Section({
  children,
  id,
  className = "",
}: {
  children: React.ReactNode;
  id: string;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-10%" });

  return (
    <section
      id={id}
      ref={ref}
      className={`min-h-screen w-full flex flex-col items-center justify-center relative snap-start px-6 md:px-12 lg:px-24 ${className}`}
    >
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.9, ease: "easeOut" }}
        className="w-full max-w-6xl"
      >
        {children}
      </motion.div>
    </section>
  );
}

// ---------- PARTICLE FIELD ----------
function Particles() {
  const particles = useMemo(
    () =>
      Array.from({ length: 40 }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: Math.random() * 2 + 0.5,
        duration: Math.random() * 20 + 15,
        delay: Math.random() * 10,
        opacity: Math.random() * 0.3 + 0.05,
      })),
    []
  );

  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      {particles.map((p) => (
        <motion.div
          key={p.id}
          className="absolute rounded-full"
          style={{
            left: `${p.x}%`,
            top: `${p.y}%`,
            width: p.size,
            height: p.size,
            background: `rgba(34, 211, 238, ${p.opacity})`,
          }}
          animate={{
            y: [0, -30, 0],
            opacity: [p.opacity, p.opacity * 2, p.opacity],
          }}
          transition={{
            duration: p.duration,
            repeat: Infinity,
            delay: p.delay,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}

// ---------- RESULTS BAR CHART ----------
const moleculeResults = [
  { molecule: "H₂O", mrmse: 0.218, color: "#3b82f6" },
  { molecule: "CO₂", mrmse: 0.261, color: "#22d3ee" },
  { molecule: "CH₄", mrmse: 0.29, color: "#eab308" },
  { molecule: "CO", mrmse: 0.327, color: "#10b981" },
  { molecule: "NH₃", mrmse: 0.378, color: "#f97316" },
];

const comparisonData = [
  { name: "Random Forest", mrmse: 1.2, color: "#374151" },
  { name: "CNN Baseline", mrmse: 0.85, color: "#4b5563" },
  { name: "ADC 2023 Winner", mrmse: 0.32, color: "#6b7280" },
  { name: "ExoBiome", mrmse: 0.295, color: "#22d3ee" },
];

// Molecule info for the per-molecule section
const moleculeDetails = [
  {
    name: "H₂O",
    fullName: "Water",
    mrmse: 0.218,
    color: "#3b82f6",
    primaryWl: 2.7,
    note: "Strongest retrieval — key habitability marker",
  },
  {
    name: "CO₂",
    fullName: "Carbon Dioxide",
    mrmse: 0.261,
    color: "#22d3ee",
    primaryWl: 4.3,
    note: "Deep 4.3μm band clearly resolved",
  },
  {
    name: "CH₄",
    fullName: "Methane",
    mrmse: 0.29,
    color: "#eab308",
    primaryWl: 3.3,
    note: "Biosignature gas — biogenic vs. abiogenic",
  },
  {
    name: "CO",
    fullName: "Carbon Monoxide",
    mrmse: 0.327,
    color: "#10b981",
    primaryWl: 4.7,
    note: "Disequilibrium indicator with CO₂",
  },
  {
    name: "NH₃",
    fullName: "Ammonia",
    mrmse: 0.378,
    color: "#f97316",
    primaryWl: 6.1,
    note: "Hardest target — weak overlapping features",
  },
];

// ---------- ARCHITECTURE DIAGRAM ----------
function ArchitectureDiagram() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-15%" });

  const blocks = [
    { label: "Spectrum\n52 × 1", sub: "0.5–7.8 μm", color: "#3b82f6", delay: 0 },
    { label: "Spectral\nEncoder", sub: "Conv1D + GELU", color: "#6366f1", delay: 0.1 },
    { label: "Aux\nEncoder", sub: "Rₚ, Tₛ, [Fe/H]...", color: "#8b5cf6", delay: 0.15 },
    { label: "Fusion\nLayer", sub: "Concat + Dense", color: "#a855f7", delay: 0.2 },
    { label: "Quantum\nCircuit", sub: "12 qubits", color: "#22d3ee", delay: 0.3 },
    { label: "5 Molecules\nlog₁₀ VMR", sub: "Output", color: "#10b981", delay: 0.4 },
  ];

  return (
    <div ref={ref} className="w-full overflow-x-auto">
      <div className="flex items-center justify-center gap-2 md:gap-4 min-w-[700px] py-8">
        {blocks.map((block, i) => (
          <div key={i} className="flex items-center gap-2 md:gap-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={inView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.5, delay: block.delay }}
              className="flex flex-col items-center"
            >
              <div
                className="relative px-4 py-5 rounded-xl border text-center min-w-[100px]"
                style={{
                  borderColor: block.color + "60",
                  background: block.color + "10",
                  boxShadow: `0 0 30px ${block.color}15`,
                }}
              >
                <div
                  className="text-xs md:text-sm font-medium whitespace-pre-line leading-tight"
                  style={{ color: block.color }}
                >
                  {block.label}
                </div>
                <div className="text-[10px] text-gray-500 mt-1 font-mono">
                  {block.sub}
                </div>
                {block.label.includes("Quantum") && (
                  <motion.div
                    className="absolute -inset-[1px] rounded-xl border"
                    style={{ borderColor: block.color }}
                    animate={{ opacity: [0.3, 0.8, 0.3] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                )}
              </div>
            </motion.div>
            {i < blocks.length - 1 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={inView ? { opacity: 0.5 } : {}}
                transition={{ duration: 0.5, delay: block.delay + 0.2 }}
              >
                <svg width="24" height="12" viewBox="0 0 24 12">
                  <path
                    d="M0 6 L18 6 M14 2 L18 6 L14 10"
                    stroke="#4b5563"
                    strokeWidth="1.5"
                    fill="none"
                  />
                </svg>
              </motion.div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------- NAVIGATION ----------
function NavDot({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group relative flex items-center"
      aria-label={label}
    >
      <span
        className={`block w-2.5 h-2.5 rounded-full border transition-all duration-300 ${
          active
            ? "bg-cyan-400 border-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]"
            : "bg-transparent border-gray-600 group-hover:border-gray-400"
        }`}
      />
      <span
        className={`absolute right-6 text-xs font-mono whitespace-nowrap transition-opacity duration-200 ${
          active ? "text-cyan-400 opacity-100" : "text-gray-500 opacity-0 group-hover:opacity-100"
        }`}
      >
        {label}
      </span>
    </button>
  );
}

// ---------- CUSTOM TOOLTIP ----------
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-gray-900/95 border border-gray-700 rounded-lg px-3 py-2 text-xs font-mono shadow-xl backdrop-blur-sm">
      <p className="text-gray-300">{label || payload[0]?.payload?.name || payload[0]?.payload?.molecule}</p>
      <p className="text-cyan-400 font-medium">
        mRMSE: {payload[0].value.toFixed(3)}
      </p>
    </div>
  );
}

// ==================== MAIN PAGE ====================
export default function ExoBiomePage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [breathePhase, setBreathePhase] = useState(0);
  const [activeSection, setActiveSection] = useState(0);

  const sectionIds = [
    "hero",
    "problem",
    "architecture",
    "results",
    "molecules",
    "comparison",
    "team",
  ];

  const sectionLabels = [
    "Home",
    "The Problem",
    "Architecture",
    "Results",
    "Molecules",
    "Comparison",
    "Team",
  ];

  // Breathing animation for spectrum
  useEffect(() => {
    let raf: number;
    const animate = () => {
      setBreathePhase((p) => p + 0.008);
      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Track scroll position for active section
  const { scrollYProgress } = useScroll({ container: containerRef });
  const smoothProgress = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
  });

  useEffect(() => {
    const unsub = smoothProgress.on("change", (v) => {
      const idx = Math.round(v * (sectionIds.length - 1));
      setActiveSection(Math.min(idx, sectionIds.length - 1));
    });
    return unsub;
  }, [smoothProgress, sectionIds.length]);

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    el?.scrollIntoView({ behavior: "smooth" });
  };

  // Hero spectrum opacity fades as user scrolls
  const heroSpectrumOpacity = useTransform(scrollYProgress, [0, 0.15], [1, 0]);

  return (
    <div
      className={`${dmSans.variable} ${jetbrains.variable} bg-[#060612] text-white font-[family-name:var(--font-dm-sans)]`}
    >
      <Particles />

      {/* Fixed nav dots */}
      <nav className="fixed right-6 top-1/2 -translate-y-1/2 z-50 flex flex-col gap-4 hidden md:flex">
        {sectionIds.map((id, i) => (
          <NavDot
            key={id}
            active={activeSection === i}
            label={sectionLabels[i]}
            onClick={() => scrollTo(id)}
          />
        ))}
      </nav>

      {/* Fixed spectrum in background for hero */}
      <motion.div
        className="fixed top-1/2 left-0 w-full -translate-y-1/2 z-[1] pointer-events-none"
        style={{ opacity: heroSpectrumOpacity }}
      >
        <SpectrumSVG
          breathePhase={breathePhase}
          glowIntensity={1.2}
          showLabels={true}
          className="w-full"
        />
      </motion.div>

      {/* Scroll container */}
      <div
        ref={containerRef}
        className="relative z-10 h-screen overflow-y-auto snap-y snap-mandatory scroll-smooth"
      >
        {/* ===== HERO ===== */}
        <section
          id="hero"
          className="min-h-screen w-full flex flex-col items-center justify-center relative snap-start px-6"
        >
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1.5 }}
            className="text-center relative z-10"
          >
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.8 }}
              className="text-sm md:text-base font-mono tracking-[0.3em] text-cyan-400/70 uppercase mb-6"
            >
              HACK-4-SAGES 2026 &middot; ETH Zurich
            </motion.p>

            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 1 }}
              className="text-5xl md:text-7xl lg:text-8xl font-light tracking-tight mb-4"
            >
              <span className="text-white">Exo</span>
              <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-violet-400 bg-clip-text text-transparent font-semibold">
                Biome
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1, duration: 0.8 }}
              className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed font-light"
            >
              Quantum-enhanced detection of biosignatures
              <br />
              from exoplanet transmission spectra
            </motion.p>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.6, duration: 1 }}
              className="mt-12 flex items-center gap-8 justify-center text-xs font-mono text-gray-500"
            >
              <span>52 wavelength bins</span>
              <span className="w-px h-4 bg-gray-700" />
              <span>12 qubits</span>
              <span className="w-px h-4 bg-gray-700" />
              <span>5 molecules</span>
              <span className="w-px h-4 bg-gray-700" />
              <span>0.295 mRMSE</span>
            </motion.div>
          </motion.div>

          {/* Scroll indicator */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: [0, 0.6, 0] }}
            transition={{ delay: 2.5, duration: 2, repeat: Infinity }}
            className="absolute bottom-10 left-1/2 -translate-x-1/2"
          >
            <div className="flex flex-col items-center gap-2 text-gray-500">
              <span className="text-xs font-mono">scroll</span>
              <svg width="16" height="24" viewBox="0 0 16 24" fill="none">
                <path
                  d="M8 4 L8 18 M4 14 L8 18 L12 14"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
              </svg>
            </div>
          </motion.div>
        </section>

        {/* ===== THE PROBLEM ===== */}
        <Section id="problem">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <p className="text-xs font-mono tracking-[0.2em] text-cyan-400/60 uppercase mb-4">
                The Challenge
              </p>
              <h2 className="text-3xl md:text-4xl font-light mb-6 leading-tight">
                Reading the light
                <br />
                of distant worlds
              </h2>
              <p className="text-gray-400 leading-relaxed mb-6">
                When starlight passes through an exoplanet&apos;s atmosphere,
                molecules absorb specific wavelengths, leaving fingerprints in
                the transmission spectrum. Retrieving molecular abundances from
                these spectra is an inverse problem — classically solved with
                Bayesian methods requiring hours of compute per target.
              </p>
              <p className="text-gray-400 leading-relaxed">
                ExoBiome replaces this pipeline with a quantum-classical neural
                network that predicts log₁₀ volume mixing ratios for five key
                molecules in under a second.
              </p>
            </div>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-[#060612] via-transparent to-[#060612] z-10 pointer-events-none" />
              <SpectrumSVG
                breathePhase={breathePhase * 0.5}
                glowIntensity={0.8}
                showLabels={true}
                className="w-full"
              />
              <div className="text-center mt-4 text-xs font-mono text-gray-600">
                Synthetic transmission spectrum — 52 wavelength bins (0.5–7.8 μm)
              </div>
            </div>
          </div>
        </Section>

        {/* ===== ARCHITECTURE ===== */}
        <Section id="architecture">
          <p className="text-xs font-mono tracking-[0.2em] text-cyan-400/60 uppercase mb-4 text-center">
            Architecture
          </p>
          <h2 className="text-3xl md:text-4xl font-light mb-4 text-center">
            Hybrid quantum-classical pipeline
          </h2>
          <p className="text-gray-500 text-center max-w-2xl mx-auto mb-10 text-sm">
            Spectral features and auxiliary parameters are fused classically,
            then projected through a parameterized quantum circuit on 12 qubits.
          </p>

          <ArchitectureDiagram />

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-12">
            {[
              { label: "Qubits", value: "12", sub: "Parameterized circuit" },
              { label: "Input bins", value: "52", sub: "0.5–7.8 μm" },
              { label: "Training set", value: "41,423", sub: "ADC 2023 spectra" },
              { label: "Outputs", value: "5", sub: "Molecular abundances" },
            ].map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1, duration: 0.5 }}
                viewport={{ once: true }}
                className="text-center p-4 rounded-xl border border-gray-800/50 bg-gray-900/20"
              >
                <div className="text-2xl md:text-3xl font-mono font-light text-white mb-1">
                  {stat.value}
                </div>
                <div className="text-xs font-mono text-cyan-400/70">{stat.label}</div>
                <div className="text-[10px] text-gray-600 mt-1">{stat.sub}</div>
              </motion.div>
            ))}
          </div>
        </Section>

        {/* ===== RESULTS ===== */}
        <Section id="results">
          <div className="text-center mb-12">
            <p className="text-xs font-mono tracking-[0.2em] text-cyan-400/60 uppercase mb-4">
              Performance
            </p>
            <h2 className="text-3xl md:text-4xl font-light mb-6">
              State-of-the-art retrieval accuracy
            </h2>

            <div className="relative inline-block">
              {/* Pulsing glow behind the number */}
              <motion.div
                className="absolute inset-0 rounded-full"
                style={{
                  background:
                    "radial-gradient(ellipse at center, rgba(34,211,238,0.15) 0%, transparent 70%)",
                }}
                animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0.8, 0.5] }}
                transition={{ duration: 3, repeat: Infinity }}
              />
              <div className="relative font-mono text-7xl md:text-9xl font-light text-white tracking-tight">
                <AnimatedNumber value={0.295} decimals={3} duration={2.5} />
              </div>
              <div className="text-sm font-mono text-gray-500 mt-2">
                mean Root Mean Square Error
              </div>
            </div>
          </div>

          {/* Per-molecule bar chart */}
          <div className="max-w-xl mx-auto h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={moleculeResults}
                margin={{ top: 10, right: 10, left: -10, bottom: 5 }}
                barCategoryGap="25%"
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis
                  dataKey="molecule"
                  tick={{ fill: "#9ca3af", fontSize: 12, fontFamily: "var(--font-jetbrains)" }}
                  axisLine={{ stroke: "#1f2937" }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: "#6b7280", fontSize: 10, fontFamily: "var(--font-jetbrains)" }}
                  axisLine={false}
                  tickLine={false}
                  domain={[0, 0.5]}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(34,211,238,0.05)" }} />
                <Bar dataKey="mrmse" radius={[4, 4, 0, 0]} animationDuration={1500}>
                  {moleculeResults.map((entry, i) => (
                    <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Section>

        {/* ===== PER-MOLECULE ===== */}
        <Section id="molecules">
          <p className="text-xs font-mono tracking-[0.2em] text-cyan-400/60 uppercase mb-4 text-center">
            Molecular Retrieval
          </p>
          <h2 className="text-3xl md:text-4xl font-light mb-10 text-center">
            Five molecules, one spectrum
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {moleculeDetails.map((mol, i) => (
              <motion.div
                key={mol.name}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08, duration: 0.5 }}
                viewport={{ once: true }}
                className="rounded-xl border border-gray-800/50 bg-gray-900/30 p-4 hover:border-gray-700 transition-colors group"
              >
                <div className="flex items-baseline gap-2 mb-1">
                  <span
                    className="text-xl font-mono font-semibold"
                    style={{ color: mol.color }}
                  >
                    {mol.name}
                  </span>
                  <span className="text-[10px] text-gray-600">{mol.fullName}</span>
                </div>

                <MoleculeSpectrum
                  molecule={mol.name}
                  color={mol.color}
                  primaryWl={mol.primaryWl}
                />

                <div className="mt-2 flex items-baseline gap-1.5">
                  <span className="text-2xl font-mono font-light text-white">
                    {mol.mrmse.toFixed(3)}
                  </span>
                  <span className="text-[10px] text-gray-600 font-mono">mRMSE</span>
                </div>

                <p className="text-[11px] text-gray-500 mt-1 leading-snug">
                  {mol.note}
                </p>

                <div className="flex items-center gap-1 mt-2">
                  <span className="text-[9px] font-mono text-gray-600">
                    Primary: {mol.primaryWl}μm
                  </span>
                </div>
              </motion.div>
            ))}
          </div>
        </Section>

        {/* ===== COMPARISON ===== */}
        <Section id="comparison">
          <p className="text-xs font-mono tracking-[0.2em] text-cyan-400/60 uppercase mb-4 text-center">
            Benchmarks
          </p>
          <h2 className="text-3xl md:text-4xl font-light mb-4 text-center">
            Beating the best
          </h2>
          <p className="text-gray-500 text-center max-w-lg mx-auto mb-10 text-sm">
            ExoBiome outperforms the ADC 2023 winning solution and
            classical baselines on mean RMSE across all five target molecules.
          </p>

          <div className="max-w-2xl mx-auto">
            {comparisonData.map((item, i) => {
              const isExoBiome = item.name === "ExoBiome";
              const barWidth = ((1.3 - item.mrmse) / 1.3) * 100;
              return (
                <motion.div
                  key={item.name}
                  initial={{ opacity: 0, x: -30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.12, duration: 0.5 }}
                  viewport={{ once: true }}
                  className="mb-4"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span
                      className={`text-sm font-mono ${
                        isExoBiome ? "text-cyan-400 font-medium" : "text-gray-400"
                      }`}
                    >
                      {item.name}
                    </span>
                    <span
                      className={`text-sm font-mono ${
                        isExoBiome ? "text-white font-semibold" : "text-gray-500"
                      }`}
                    >
                      {item.mrmse.toFixed(3)}
                    </span>
                  </div>
                  <div className="h-3 w-full bg-gray-900 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{
                        background: isExoBiome
                          ? "linear-gradient(90deg, #22d3ee, #3b82f6)"
                          : item.color,
                      }}
                      initial={{ width: 0 }}
                      whileInView={{ width: `${barWidth}%` }}
                      transition={{ duration: 1, delay: i * 0.12 + 0.2 }}
                      viewport={{ once: true }}
                    />
                  </div>
                </motion.div>
              );
            })}

            <div className="text-center mt-8 text-xs font-mono text-gray-600">
              Lower mRMSE is better &middot; Bars show relative accuracy (inverted scale)
            </div>
          </div>
        </Section>

        {/* ===== TEAM / CLOSING ===== */}
        <Section id="team">
          <div className="text-center">
            <p className="text-xs font-mono tracking-[0.2em] text-cyan-400/60 uppercase mb-4">
              The Team
            </p>
            <h2 className="text-3xl md:text-4xl font-light mb-6">
              Built at HACK-4-SAGES 2026
            </h2>
            <p className="text-gray-400 max-w-lg mx-auto mb-10 leading-relaxed text-sm">
              A four-person team combining quantum computing, machine learning,
              and astrophysics. Three working days. One question:
            </p>
            <p className="text-lg md:text-xl text-gray-300 italic font-light mb-12">
              &quot;Can quantum computers help us find life beyond Earth?&quot;
            </p>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-2xl mx-auto mb-16">
              {[
                { name: "Quantum ML", icon: "◈" },
                { name: "Spectroscopy", icon: "◇" },
                { name: "Astrophysics", icon: "☉" },
                { name: "Data Science", icon: "▦" },
              ].map((role, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1, duration: 0.5 }}
                  viewport={{ once: true }}
                  className="text-center"
                >
                  <div className="text-2xl mb-2 text-cyan-400/60">{role.icon}</div>
                  <div className="text-xs font-mono text-gray-400">{role.name}</div>
                </motion.div>
              ))}
            </div>

            {/* Closing spectrum */}
            <div className="relative max-w-3xl mx-auto">
              <div className="absolute inset-0 bg-gradient-to-r from-[#060612] via-transparent to-[#060612] z-10 pointer-events-none" />
              <SpectrumSVG
                breathePhase={breathePhase * 0.3}
                opacity={0.4}
                glowIntensity={0.5}
                showLabels={false}
                className="w-full"
              />
            </div>

            <div className="mt-8 flex flex-wrap items-center justify-center gap-6 text-xs font-mono text-gray-600">
              <span>Dataset: Ariel Data Challenge 2023</span>
              <span className="w-px h-3 bg-gray-800" />
              <span>Hardware: IQM Spark (5 qubits) &middot; VTT Q50 (53 qubits)</span>
              <span className="w-px h-3 bg-gray-800" />
              <span>Category: Life Detection &amp; Biosignatures</span>
            </div>

            <div className="mt-6 text-[10px] font-mono text-gray-700">
              ETH Zurich &middot; COPL &middot; Origins Federation Conference
            </div>
          </div>
        </Section>
      </div>
    </div>
  );
}
