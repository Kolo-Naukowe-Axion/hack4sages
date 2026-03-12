"use client";

import { useRef, useEffect, useState, useMemo } from "react";
import { motion, useInView, AnimatePresence } from "framer-motion";
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
  Radar,
  Legend,
} from "recharts";
import { Playfair_Display, JetBrains_Mono } from "next/font/google";

const playfair = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-playfair",
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

// ─── Floating Orb Background ─────────────────────────────────────────────────

function FloatingOrbs() {
  const orbs = useMemo(
    () => [
      {
        size: 600,
        x: "10%",
        y: "20%",
        color: "rgba(99, 102, 241, 0.04)",
        duration: 25,
        delay: 0,
      },
      {
        size: 500,
        x: "70%",
        y: "60%",
        color: "rgba(168, 85, 247, 0.03)",
        duration: 30,
        delay: 2,
      },
      {
        size: 400,
        x: "40%",
        y: "80%",
        color: "rgba(59, 130, 246, 0.035)",
        duration: 22,
        delay: 5,
      },
      {
        size: 350,
        x: "80%",
        y: "15%",
        color: "rgba(16, 185, 129, 0.025)",
        duration: 28,
        delay: 1,
      },
      {
        size: 450,
        x: "25%",
        y: "50%",
        color: "rgba(139, 92, 246, 0.03)",
        duration: 35,
        delay: 3,
      },
    ],
    []
  );

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
      {orbs.map((orb, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full"
          style={{
            width: orb.size,
            height: orb.size,
            left: orb.x,
            top: orb.y,
            background: `radial-gradient(circle, ${orb.color} 0%, transparent 70%)`,
            filter: "blur(80px)",
          }}
          animate={{
            x: [0, 40, -30, 20, 0],
            y: [0, -30, 20, -40, 0],
            scale: [1, 1.1, 0.95, 1.05, 1],
          }}
          transition={{
            duration: orb.duration,
            repeat: Infinity,
            ease: "easeInOut",
            delay: orb.delay,
          }}
        />
      ))}
    </div>
  );
}

// ─── Animated Counter ────────────────────────────────────────────────────────

function AnimatedNumber({
  value,
  decimals = 3,
  prefix = "",
  suffix = "",
  duration = 2,
}: {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  duration?: number;
}) {
  const [display, setDisplay] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-50px" });

  useEffect(() => {
    if (!inView) return;
    const start = performance.now();
    const step = (now: number) => {
      const elapsed = (now - start) / 1000;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(value * eased);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [inView, value, duration]);

  return (
    <span ref={ref} className={jetbrains.className}>
      {prefix}
      {display.toFixed(decimals)}
      {suffix}
    </span>
  );
}

// ─── Slide Wrapper ───────────────────────────────────────────────────────────

function Slide({
  left,
  right,
  index,
}: {
  left: React.ReactNode;
  right: React.ReactNode;
  index: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section
      ref={ref}
      className="h-screen w-full snap-start snap-always relative flex"
      style={{ scrollSnapAlign: "start" }}
    >
      {/* Left panel — narrative */}
      <motion.div
        className="w-1/2 h-full flex items-center justify-center px-12 lg:px-20 relative z-10"
        style={{ backgroundColor: "#080808" }}
        initial={{ x: -80, opacity: 0 }}
        animate={inView ? { x: 0, opacity: 1 } : {}}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1], delay: 0.1 }}
      >
        <div className="max-w-lg">{left}</div>
      </motion.div>

      {/* Divider */}
      <div
        className="w-px h-full relative z-10"
        style={{ backgroundColor: "rgba(255,255,255,0.07)" }}
      />

      {/* Right panel — data */}
      <motion.div
        className="w-1/2 h-full flex items-center justify-center px-10 lg:px-16 relative z-10"
        style={{ backgroundColor: "#0e0e0e" }}
        initial={{ x: 80, opacity: 0 }}
        animate={inView ? { x: 0, opacity: 1 } : {}}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1], delay: 0.25 }}
      >
        <div className="w-full max-w-xl">{right}</div>
      </motion.div>

      {/* Slide indicator */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-20">
        <span
          className={`text-xs tracking-[0.3em] uppercase ${jetbrains.className}`}
          style={{ color: "rgba(255,255,255,0.15)" }}
        >
          {String(index + 1).padStart(2, "0")} / 07
        </span>
      </div>
    </section>
  );
}

// ─── Molecule Badge ──────────────────────────────────────────────────────────

const moleculeColors: Record<string, string> = {
  "H₂O": "#3b82f6",
  "CO₂": "#f59e0b",
  CO: "#ef4444",
  "CH₄": "#10b981",
  "NH₃": "#a855f7",
};

const moleculeWavelengths: Record<string, string> = {
  "H₂O": "1.4, 1.9, 2.7 μm",
  "CO₂": "2.0, 4.3 μm",
  CO: "2.3, 4.6 μm",
  "CH₄": "1.7, 3.3 μm",
  "NH₃": "1.5, 2.0, 3.0 μm",
};

function MoleculeBadge({
  name,
  delay,
}: {
  name: string;
  delay: number;
}) {
  const color = moleculeColors[name] || "#888";
  return (
    <motion.div
      className="flex items-center gap-4 p-4 rounded-lg border"
      style={{
        borderColor: `${color}22`,
        backgroundColor: `${color}08`,
      }}
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: delay * 0.12, duration: 0.6 }}
    >
      <div
        className="w-3 h-3 rounded-full flex-shrink-0"
        style={{ backgroundColor: color, boxShadow: `0 0 12px ${color}44` }}
      />
      <div>
        <p className={`text-lg font-medium text-white ${jetbrains.className}`}>
          {name}
        </p>
        <p
          className={`text-xs mt-0.5 ${jetbrains.className}`}
          style={{ color: "rgba(255,255,255,0.35)" }}
        >
          {moleculeWavelengths[name]}
        </p>
      </div>
    </motion.div>
  );
}

// ─── Architecture Node ───────────────────────────────────────────────────────

function ArchNode({
  label,
  sub,
  color,
  delay,
}: {
  label: string;
  sub: string;
  color: string;
  delay: number;
}) {
  return (
    <motion.div
      className="relative px-5 py-3.5 rounded-lg border text-center"
      style={{
        borderColor: `${color}33`,
        backgroundColor: `${color}0a`,
      }}
      initial={{ opacity: 0, scale: 0.85 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      transition={{ delay, duration: 0.5 }}
    >
      <p className={`text-sm font-medium text-white ${jetbrains.className}`}>
        {label}
      </p>
      <p
        className={`text-[10px] mt-1 ${jetbrains.className}`}
        style={{ color: "rgba(255,255,255,0.3)" }}
      >
        {sub}
      </p>
    </motion.div>
  );
}

function ArchArrow({ delay }: { delay: number }) {
  return (
    <motion.div
      className="flex items-center justify-center py-1"
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ delay, duration: 0.4 }}
    >
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        <path
          d="M12 5v14M5 12l7 7 7-7"
          stroke="rgba(255,255,255,0.2)"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </motion.div>
  );
}

// ─── Custom Tooltip ──────────────────────────────────────────────────────────

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className={`px-4 py-2.5 rounded-lg border ${jetbrains.className}`}
      style={{
        backgroundColor: "#1a1a1a",
        borderColor: "rgba(255,255,255,0.1)",
      }}
    >
      <p className="text-xs text-white/50 mb-1">{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} className="text-sm" style={{ color: p.color }}>
          {p.name}: {p.value.toFixed(3)}
        </p>
      ))}
    </div>
  );
}

// ─── Stat Card ───────────────────────────────────────────────────────────────

function StatCard({
  label,
  value,
  sub,
  delay,
}: {
  label: string;
  value: string;
  sub?: string;
  delay: number;
}) {
  return (
    <motion.div
      className="p-5 rounded-lg border"
      style={{
        borderColor: "rgba(255,255,255,0.06)",
        backgroundColor: "rgba(255,255,255,0.02)",
      }}
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay, duration: 0.6 }}
    >
      <p
        className={`text-xs uppercase tracking-[0.15em] mb-2 ${jetbrains.className}`}
        style={{ color: "rgba(255,255,255,0.3)" }}
      >
        {label}
      </p>
      <p className={`text-2xl font-semibold text-white ${jetbrains.className}`}>
        {value}
      </p>
      {sub && (
        <p
          className={`text-xs mt-1 ${jetbrains.className}`}
          style={{ color: "rgba(255,255,255,0.25)" }}
        >
          {sub}
        </p>
      )}
    </motion.div>
  );
}

// ─── Particle Field ──────────────────────────────────────────────────────────

function ParticleField() {
  const particles = useMemo(() => {
    return Array.from({ length: 60 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 2 + 0.5,
      duration: Math.random() * 20 + 15,
      delay: Math.random() * 10,
      opacity: Math.random() * 0.15 + 0.05,
    }));
  }, []);

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
            backgroundColor: `rgba(255,255,255,${p.opacity})`,
          }}
          animate={{
            y: [0, -30, 10, -20, 0],
            x: [0, 15, -10, 5, 0],
            opacity: [p.opacity, p.opacity * 1.5, p.opacity * 0.5, p.opacity],
          }}
          transition={{
            duration: p.duration,
            repeat: Infinity,
            ease: "easeInOut",
            delay: p.delay,
          }}
        />
      ))}
    </div>
  );
}

// ─── Scroll Progress ─────────────────────────────────────────────────────────

function ScrollProgress() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const container = document.getElementById("scroll-container");
    if (!container) return;
    const handler = () => {
      const scrollTop = container.scrollTop;
      const scrollHeight = container.scrollHeight - container.clientHeight;
      setProgress(scrollHeight > 0 ? scrollTop / scrollHeight : 0);
    };
    container.addEventListener("scroll", handler, { passive: true });
    return () => container.removeEventListener("scroll", handler);
  }, []);

  return (
    <div
      className="fixed top-0 left-0 h-[2px] z-50"
      style={{
        width: `${progress * 100}%`,
        background:
          "linear-gradient(90deg, #6366f1, #a855f7, #3b82f6)",
        transition: "width 0.1s ease-out",
      }}
    />
  );
}

// ─── Nav Dots ────────────────────────────────────────────────────────────────

function NavDots({ activeSlide }: { activeSlide: number }) {
  return (
    <div className="fixed right-6 top-1/2 -translate-y-1/2 z-50 flex flex-col gap-3">
      {Array.from({ length: 7 }, (_, i) => (
        <button
          key={i}
          onClick={() => {
            const container = document.getElementById("scroll-container");
            if (container) {
              container.scrollTo({
                top: i * window.innerHeight,
                behavior: "smooth",
              });
            }
          }}
          className="group relative flex items-center justify-center"
        >
          <motion.div
            className="rounded-full"
            style={{
              width: activeSlide === i ? 8 : 4,
              height: activeSlide === i ? 8 : 4,
              backgroundColor:
                activeSlide === i
                  ? "rgba(255,255,255,0.7)"
                  : "rgba(255,255,255,0.15)",
            }}
            animate={{
              scale: activeSlide === i ? 1 : 0.8,
            }}
            transition={{ duration: 0.3 }}
          />
        </button>
      ))}
    </div>
  );
}

// ─── Data ────────────────────────────────────────────────────────────────────

const comparisonData = [
  { name: "Random Forest", mrmse: 1.2, fill: "#374151" },
  { name: "CNN Baseline", mrmse: 0.85, fill: "#4b5563" },
  { name: "ADC 2023 Winner", mrmse: 0.32, fill: "#6366f1" },
  { name: "ExoBiome", mrmse: 0.295, fill: "#a855f7" },
];

const perMoleculeData = [
  { molecule: "H₂O", exobiome: 0.218, adc: 0.32, color: "#3b82f6" },
  { molecule: "CO₂", exobiome: 0.261, adc: 0.32, color: "#f59e0b" },
  { molecule: "CO", exobiome: 0.327, adc: 0.32, color: "#ef4444" },
  { molecule: "CH₄", exobiome: 0.290, adc: 0.32, color: "#10b981" },
  { molecule: "NH₃", exobiome: 0.378, adc: 0.32, color: "#a855f7" },
];

const radarData = [
  { metric: "H₂O", value: (1 - 0.218) * 100, fullMark: 100 },
  { metric: "CO₂", value: (1 - 0.261) * 100, fullMark: 100 },
  { metric: "CO", value: (1 - 0.327) * 100, fullMark: 100 },
  { metric: "CH₄", value: (1 - 0.290) * 100, fullMark: 100 },
  { metric: "NH₃", value: (1 - 0.378) * 100, fullMark: 100 },
];

// ─── Main Page ───────────────────────────────────────────────────────────────

export default function ExoBiomePage() {
  const [activeSlide, setActiveSlide] = useState(0);

  useEffect(() => {
    const container = document.getElementById("scroll-container");
    if (!container) return;
    const handler = () => {
      const index = Math.round(container.scrollTop / window.innerHeight);
      setActiveSlide(Math.min(index, 6));
    };
    container.addEventListener("scroll", handler, { passive: true });
    return () => container.removeEventListener("scroll", handler);
  }, []);

  return (
    <div
      className={`${playfair.variable} ${jetbrains.variable} bg-[#080808] text-white min-h-screen`}
    >
      <FloatingOrbs />
      <ParticleField />
      <ScrollProgress />
      <NavDots activeSlide={activeSlide} />

      <div
        id="scroll-container"
        className="h-screen overflow-y-auto"
        style={{
          scrollSnapType: "y mandatory",
          scrollBehavior: "smooth",
        }}
      >
        {/* ── Slide 1: Hero ─────────────────────────────────────────────── */}
        <Slide
          index={0}
          left={
            <div>
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6 }}
              >
                <p
                  className={`text-xs tracking-[0.3em] uppercase mb-8 ${jetbrains.className}`}
                  style={{ color: "rgba(255,255,255,0.3)" }}
                >
                  HACK-4-SAGES 2026 &middot; ETH Zurich
                </p>
              </motion.div>

              <motion.h1
                className={`text-6xl lg:text-7xl font-light leading-[1.1] mb-6 ${playfair.className}`}
                style={{
                  background:
                    "linear-gradient(135deg, #ffffff 0%, rgba(255,255,255,0.6) 100%)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                }}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: 0.15 }}
              >
                Exo
                <br />
                Biome
              </motion.h1>

              <motion.p
                className={`text-lg leading-relaxed mb-6 ${playfair.className}`}
                style={{ color: "rgba(255,255,255,0.5)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: 0.3 }}
              >
                Quantum-enhanced biosignature detection from exoplanet
                transmission spectra. The first application of quantum machine
                learning to identify molecular signatures of life beyond Earth.
              </motion.p>

              <motion.div
                className="flex items-center gap-3"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.5 }}
              >
                <div
                  className="w-8 h-px"
                  style={{ backgroundColor: "rgba(168,85,247,0.5)" }}
                />
                <span
                  className={`text-xs tracking-wider ${jetbrains.className}`}
                  style={{ color: "rgba(255,255,255,0.25)" }}
                >
                  Scroll to explore
                </span>
              </motion.div>
            </div>
          }
          right={
            <div className="space-y-8">
              <motion.div
                className="text-center mb-10"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4, duration: 0.8 }}
              >
                <p
                  className={`text-[80px] lg:text-[96px] font-light leading-none ${jetbrains.className}`}
                  style={{
                    background:
                      "linear-gradient(135deg, #a855f7, #6366f1)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                  }}
                >
                  <AnimatedNumber value={0.295} decimals={3} />
                </p>
                <p
                  className={`text-xs tracking-[0.2em] uppercase mt-3 ${jetbrains.className}`}
                  style={{ color: "rgba(255,255,255,0.3)" }}
                >
                  Mean RMSE
                </p>
              </motion.div>

              <div className="grid grid-cols-3 gap-4">
                <StatCard
                  label="Qubits"
                  value="12"
                  sub="Quantum circuit"
                  delay={0.5}
                />
                <StatCard
                  label="Molecules"
                  value="5"
                  sub="Detected"
                  delay={0.6}
                />
                <StatCard
                  label="Spectra"
                  value="41.4K"
                  sub="Training set"
                  delay={0.7}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <StatCard
                  label="Parameters"
                  value="120K"
                  sub="Trainable"
                  delay={0.8}
                />
                <StatCard
                  label="Training"
                  value="~3 min"
                  sub="Wall-clock time"
                  delay={0.9}
                />
              </div>
            </div>
          }
        />

        {/* ── Slide 2: The Challenge ───────────────────────────────────── */}
        <Slide
          index={1}
          left={
            <div>
              <motion.p
                className={`text-xs tracking-[0.3em] uppercase mb-6 ${jetbrains.className}`}
                style={{ color: "rgba(168,85,247,0.5)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                01 &mdash; The Challenge
              </motion.p>

              <motion.h2
                className={`text-4xl lg:text-5xl font-light leading-tight mb-8 ${playfair.className}`}
                style={{ color: "rgba(255,255,255,0.9)" }}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1, duration: 0.7 }}
              >
                Reading the
                <br />
                chemistry of
                <br />
                distant worlds
              </motion.h2>

              <motion.p
                className="text-base leading-relaxed mb-5"
                style={{ color: "rgba(255,255,255,0.4)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.25, duration: 0.7 }}
              >
                When starlight passes through an exoplanet&apos;s atmosphere,
                specific molecules absorb light at characteristic wavelengths.
                These absorption features encode the atmospheric composition
                &mdash; a fingerprint of the world&apos;s chemistry.
              </motion.p>

              <motion.p
                className="text-base leading-relaxed"
                style={{ color: "rgba(255,255,255,0.4)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.35, duration: 0.7 }}
              >
                The task: recover the log&#8322; volume mixing ratios of five
                key biosignature molecules from noisy transit spectra. Each
                spectrum contains 52 wavelength bins spanning 0.5&ndash;7.8 μm,
                plus auxiliary planetary parameters.
              </motion.p>
            </div>
          }
          right={
            <div className="space-y-4">
              <motion.p
                className={`text-xs tracking-[0.2em] uppercase mb-6 ${jetbrains.className}`}
                style={{ color: "rgba(255,255,255,0.2)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                Target Molecules
              </motion.p>

              {["H₂O", "CO₂", "CO", "CH₄", "NH₃"].map((mol, i) => (
                <MoleculeBadge key={mol} name={mol} delay={i} />
              ))}

              <motion.div
                className={`mt-8 p-4 rounded-lg border ${jetbrains.className}`}
                style={{
                  borderColor: "rgba(255,255,255,0.06)",
                  backgroundColor: "rgba(255,255,255,0.02)",
                }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.7, duration: 0.6 }}
              >
                <p
                  className="text-[10px] uppercase tracking-wider mb-2"
                  style={{ color: "rgba(255,255,255,0.25)" }}
                >
                  Spectral Range
                </p>
                <p className="text-sm text-white/60">
                  52 bins &middot; 0.5 &ndash; 7.8 μm &middot; Ariel
                  Telescope
                </p>
              </motion.div>
            </div>
          }
        />

        {/* ── Slide 3: Architecture ────────────────────────────────────── */}
        <Slide
          index={2}
          left={
            <div>
              <motion.p
                className={`text-xs tracking-[0.3em] uppercase mb-6 ${jetbrains.className}`}
                style={{ color: "rgba(168,85,247,0.5)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                02 &mdash; Our Approach
              </motion.p>

              <motion.h2
                className={`text-4xl lg:text-5xl font-light leading-tight mb-8 ${playfair.className}`}
                style={{ color: "rgba(255,255,255,0.9)" }}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1, duration: 0.7 }}
              >
                A hybrid
                <br />
                quantum&ndash;classical
                <br />
                architecture
              </motion.h2>

              <motion.p
                className="text-base leading-relaxed mb-5"
                style={{ color: "rgba(255,255,255,0.4)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.25, duration: 0.7 }}
              >
                Two classical encoders compress spectral and auxiliary inputs
                into a shared latent space. A fusion layer merges them into a
                12-dimensional embedding.
              </motion.p>

              <motion.p
                className="text-base leading-relaxed"
                style={{ color: "rgba(255,255,255,0.4)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.35, duration: 0.7 }}
              >
                This embedding is angle-encoded onto a 12-qubit parametrized
                circuit. Expectation values from the circuit feed into a
                classical head that outputs the five molecular abundances. The
                entire pipeline trains end-to-end via gradient descent.
              </motion.p>
            </div>
          }
          right={
            <div className="flex flex-col items-center space-y-0">
              <motion.p
                className={`text-xs tracking-[0.2em] uppercase mb-6 self-start ${jetbrains.className}`}
                style={{ color: "rgba(255,255,255,0.2)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                Pipeline
              </motion.p>

              <div className="flex gap-6 items-start">
                <div className="flex flex-col items-center">
                  <ArchNode
                    label="Spectral Encoder"
                    sub="52 bins → 64d"
                    color="#3b82f6"
                    delay={0.1}
                  />
                </div>
                <div className="flex flex-col items-center">
                  <ArchNode
                    label="Aux Encoder"
                    sub="7 params → 64d"
                    color="#f59e0b"
                    delay={0.2}
                  />
                </div>
              </div>

              <ArchArrow delay={0.3} />

              <ArchNode
                label="Fusion Layer"
                sub="128d → 12d"
                color="#6366f1"
                delay={0.35}
              />

              <ArchArrow delay={0.4} />

              <motion.div
                className="relative px-8 py-5 rounded-xl border text-center"
                style={{
                  borderColor: "rgba(168,85,247,0.35)",
                  backgroundColor: "rgba(168,85,247,0.06)",
                  boxShadow: "0 0 40px rgba(168,85,247,0.08)",
                }}
                initial={{ opacity: 0, scale: 0.85 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.45, duration: 0.6 }}
              >
                <p
                  className={`text-sm font-semibold text-white ${jetbrains.className}`}
                >
                  Quantum Circuit
                </p>
                <p
                  className={`text-[10px] mt-1 ${jetbrains.className}`}
                  style={{ color: "rgba(255,255,255,0.35)" }}
                >
                  12 qubits &middot; angle encoding &middot; parametrized
                </p>
                <div className="flex justify-center gap-1 mt-2.5">
                  {Array.from({ length: 12 }, (_, i) => (
                    <motion.div
                      key={i}
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor: `rgba(168,85,247,${0.3 + i * 0.06})`,
                      }}
                      animate={{
                        opacity: [0.4, 1, 0.4],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        delay: i * 0.15,
                      }}
                    />
                  ))}
                </div>
              </motion.div>

              <ArchArrow delay={0.55} />

              <ArchNode
                label="Classical Head"
                sub="12d → 5 molecules"
                color="#10b981"
                delay={0.6}
              />

              <ArchArrow delay={0.65} />

              <motion.div
                className="flex gap-2"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.7, duration: 0.5 }}
              >
                {["H₂O", "CO₂", "CO", "CH₄", "NH₃"].map((mol) => (
                  <span
                    key={mol}
                    className={`text-[10px] px-2.5 py-1 rounded-full border ${jetbrains.className}`}
                    style={{
                      borderColor: `${moleculeColors[mol]}33`,
                      color: moleculeColors[mol],
                    }}
                  >
                    {mol}
                  </span>
                ))}
              </motion.div>
            </div>
          }
        />

        {/* ── Slide 4: Quantum Advantage ───────────────────────────────── */}
        <Slide
          index={3}
          left={
            <div>
              <motion.p
                className={`text-xs tracking-[0.3em] uppercase mb-6 ${jetbrains.className}`}
                style={{ color: "rgba(168,85,247,0.5)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                03 &mdash; The Quantum Advantage
              </motion.p>

              <motion.h2
                className={`text-4xl lg:text-5xl font-light leading-tight mb-8 ${playfair.className}`}
                style={{ color: "rgba(255,255,255,0.9)" }}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1, duration: 0.7 }}
              >
                Outperforming
                <br />
                classical
                <br />
                benchmarks
              </motion.h2>

              <motion.p
                className="text-base leading-relaxed mb-5"
                style={{ color: "rgba(255,255,255,0.4)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.25, duration: 0.7 }}
              >
                ExoBiome achieves a mean RMSE of 0.295, surpassing the Ariel
                Data Challenge 2023 winning solution (~0.32) while using only
                120K parameters and training in under 3 minutes.
              </motion.p>

              <motion.p
                className="text-base leading-relaxed"
                style={{ color: "rgba(255,255,255,0.4)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.35, duration: 0.7 }}
              >
                The quantum circuit acts as an expressive feature transformer,
                enabling the model to capture complex correlations between
                molecular abundances that classical architectures miss. Fewer
                parameters, better generalization.
              </motion.p>
            </div>
          }
          right={
            <div>
              <motion.p
                className={`text-xs tracking-[0.2em] uppercase mb-6 ${jetbrains.className}`}
                style={{ color: "rgba(255,255,255,0.2)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                mRMSE Comparison
              </motion.p>

              <motion.div
                className="w-full h-[350px]"
                initial={{ opacity: 0, x: 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3, duration: 0.8 }}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={comparisonData}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
                    barCategoryGap="30%"
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="rgba(255,255,255,0.04)"
                      horizontal={false}
                    />
                    <XAxis
                      type="number"
                      domain={[0, 1.4]}
                      tick={{
                        fill: "rgba(255,255,255,0.25)",
                        fontSize: 11,
                        fontFamily: "var(--font-jetbrains)",
                      }}
                      axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                      tickLine={false}
                    />
                    <YAxis
                      dataKey="name"
                      type="category"
                      width={120}
                      tick={{
                        fill: "rgba(255,255,255,0.4)",
                        fontSize: 11,
                        fontFamily: "var(--font-jetbrains)",
                      }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="mrmse" radius={[0, 4, 4, 0]} name="mRMSE">
                      {comparisonData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.fill}
                          fillOpacity={index === comparisonData.length - 1 ? 1 : 0.6}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </motion.div>

              <motion.div
                className={`mt-4 flex items-center gap-3 ${jetbrains.className}`}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.6, duration: 0.5 }}
              >
                <span
                  className="text-xs"
                  style={{ color: "rgba(255,255,255,0.2)" }}
                >
                  Lower is better
                </span>
                <div
                  className="flex-1 h-px"
                  style={{ backgroundColor: "rgba(255,255,255,0.06)" }}
                />
                <span className="text-xs" style={{ color: "#a855f7" }}>
                  &darr; 7.8% vs ADC Winner
                </span>
              </motion.div>
            </div>
          }
        />

        {/* ── Slide 5: Per-Molecule ────────────────────────────────────── */}
        <Slide
          index={4}
          left={
            <div>
              <motion.p
                className={`text-xs tracking-[0.3em] uppercase mb-6 ${jetbrains.className}`}
                style={{ color: "rgba(168,85,247,0.5)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                04 &mdash; Per-Molecule Accuracy
              </motion.p>

              <motion.h2
                className={`text-4xl lg:text-5xl font-light leading-tight mb-8 ${playfair.className}`}
                style={{ color: "rgba(255,255,255,0.9)" }}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1, duration: 0.7 }}
              >
                Precision
                <br />
                across all five
                <br />
                targets
              </motion.h2>

              <motion.p
                className="text-base leading-relaxed mb-5"
                style={{ color: "rgba(255,255,255,0.4)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.25, duration: 0.7 }}
              >
                ExoBiome delivers strong performance across all five molecular
                targets. Water shows the highest accuracy at 0.218 RMSE, while
                even the most challenging molecule &mdash; ammonia &mdash;
                achieves 0.378.
              </motion.p>

              <motion.div
                className="space-y-3 mt-8"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4, duration: 0.7 }}
              >
                {perMoleculeData.map((d) => (
                  <div key={d.molecule} className="flex items-center gap-3">
                    <span
                      className={`text-sm w-10 ${jetbrains.className}`}
                      style={{ color: d.color }}
                    >
                      {d.molecule}
                    </span>
                    <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ backgroundColor: d.color }}
                        initial={{ width: 0 }}
                        whileInView={{
                          width: `${(1 - d.exobiome) * 100}%`,
                        }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.5, duration: 1.2, ease: "easeOut" }}
                      />
                    </div>
                    <span
                      className={`text-xs w-12 text-right ${jetbrains.className}`}
                      style={{ color: "rgba(255,255,255,0.4)" }}
                    >
                      {d.exobiome}
                    </span>
                  </div>
                ))}
              </motion.div>
            </div>
          }
          right={
            <div>
              <motion.p
                className={`text-xs tracking-[0.2em] uppercase mb-6 ${jetbrains.className}`}
                style={{ color: "rgba(255,255,255,0.2)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                ExoBiome vs ADC Winner &mdash; RMSE by Molecule
              </motion.p>

              <motion.div
                className="w-full h-[340px]"
                initial={{ opacity: 0, x: 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3, duration: 0.8 }}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={perMoleculeData}
                    margin={{ top: 10, right: 10, left: -10, bottom: 5 }}
                    barCategoryGap="25%"
                    barGap={4}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="rgba(255,255,255,0.04)"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="molecule"
                      tick={{
                        fill: "rgba(255,255,255,0.4)",
                        fontSize: 12,
                        fontFamily: "var(--font-jetbrains)",
                      }}
                      axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[0, 0.5]}
                      tick={{
                        fill: "rgba(255,255,255,0.25)",
                        fontSize: 11,
                        fontFamily: "var(--font-jetbrains)",
                      }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar
                      dataKey="exobiome"
                      name="ExoBiome"
                      radius={[4, 4, 0, 0]}
                    >
                      {perMoleculeData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                    <Bar
                      dataKey="adc"
                      name="ADC Winner"
                      fill="rgba(255,255,255,0.12)"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </motion.div>

              <motion.div
                className="mt-6 w-full h-[220px]"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.6, duration: 0.8 }}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.06)" />
                    <PolarAngleAxis
                      dataKey="metric"
                      tick={{
                        fill: "rgba(255,255,255,0.35)",
                        fontSize: 11,
                        fontFamily: "var(--font-jetbrains)",
                      }}
                    />
                    <Radar
                      name="Accuracy %"
                      dataKey="value"
                      stroke="#a855f7"
                      fill="#a855f7"
                      fillOpacity={0.15}
                      strokeWidth={1.5}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </motion.div>
            </div>
          }
        />

        {/* ── Slide 6: Efficiency ──────────────────────────────────────── */}
        <Slide
          index={5}
          left={
            <div>
              <motion.p
                className={`text-xs tracking-[0.3em] uppercase mb-6 ${jetbrains.className}`}
                style={{ color: "rgba(168,85,247,0.5)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                05 &mdash; Efficiency
              </motion.p>

              <motion.h2
                className={`text-4xl lg:text-5xl font-light leading-tight mb-8 ${playfair.className}`}
                style={{ color: "rgba(255,255,255,0.9)" }}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1, duration: 0.7 }}
              >
                Less compute,
                <br />
                better results
              </motion.h2>

              <motion.p
                className="text-base leading-relaxed mb-5"
                style={{ color: "rgba(255,255,255,0.4)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.25, duration: 0.7 }}
              >
                Quantum circuits provide an exponentially large Hilbert space
                for feature representation, enabling extreme parameter
                efficiency. ExoBiome achieves state-of-the-art performance with
                a fraction of the parameters.
              </motion.p>

              <motion.p
                className="text-base leading-relaxed"
                style={{ color: "rgba(255,255,255,0.4)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.35, duration: 0.7 }}
              >
                Training on the full 41,423-spectrum dataset completes in
                approximately 3 minutes on standard hardware. No GPUs required.
                No hours of hyperparameter tuning. Just a well-designed
                quantum-classical hybrid.
              </motion.p>
            </div>
          }
          right={
            <div className="space-y-8">
              <motion.p
                className={`text-xs tracking-[0.2em] uppercase mb-2 ${jetbrains.className}`}
                style={{ color: "rgba(255,255,255,0.2)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                Resource Profile
              </motion.p>

              <motion.div
                className="p-8 rounded-xl border text-center"
                style={{
                  borderColor: "rgba(99,102,241,0.15)",
                  backgroundColor: "rgba(99,102,241,0.04)",
                }}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2, duration: 0.6 }}
              >
                <p
                  className={`text-5xl lg:text-6xl font-light ${jetbrains.className}`}
                  style={{ color: "#6366f1" }}
                >
                  ~3
                </p>
                <p
                  className={`text-sm mt-2 ${jetbrains.className}`}
                  style={{ color: "rgba(255,255,255,0.35)" }}
                >
                  minutes to train
                </p>
                <div
                  className="w-12 h-px mx-auto my-4"
                  style={{ backgroundColor: "rgba(255,255,255,0.08)" }}
                />
                <p
                  className={`text-xs ${jetbrains.className}`}
                  style={{ color: "rgba(255,255,255,0.2)" }}
                >
                  41,423 spectra &middot; CPU only
                </p>
              </motion.div>

              <motion.div
                className="p-8 rounded-xl border text-center"
                style={{
                  borderColor: "rgba(168,85,247,0.15)",
                  backgroundColor: "rgba(168,85,247,0.04)",
                }}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.35, duration: 0.6 }}
              >
                <p
                  className={`text-5xl lg:text-6xl font-light ${jetbrains.className}`}
                  style={{ color: "#a855f7" }}
                >
                  120K
                </p>
                <p
                  className={`text-sm mt-2 ${jetbrains.className}`}
                  style={{ color: "rgba(255,255,255,0.35)" }}
                >
                  trainable parameters
                </p>
                <div
                  className="w-12 h-px mx-auto my-4"
                  style={{ backgroundColor: "rgba(255,255,255,0.08)" }}
                />
                <p
                  className={`text-xs ${jetbrains.className}`}
                  style={{ color: "rgba(255,255,255,0.2)" }}
                >
                  12 qubits &middot; parametrized circuit
                </p>
              </motion.div>

              <motion.div
                className="grid grid-cols-2 gap-4"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.5, duration: 0.6 }}
              >
                <div
                  className="p-4 rounded-lg border text-center"
                  style={{
                    borderColor: "rgba(255,255,255,0.06)",
                    backgroundColor: "rgba(255,255,255,0.02)",
                  }}
                >
                  <p
                    className={`text-2xl font-light text-white ${jetbrains.className}`}
                  >
                    0
                  </p>
                  <p
                    className={`text-[10px] mt-1 uppercase tracking-wider ${jetbrains.className}`}
                    style={{ color: "rgba(255,255,255,0.25)" }}
                  >
                    GPUs needed
                  </p>
                </div>
                <div
                  className="p-4 rounded-lg border text-center"
                  style={{
                    borderColor: "rgba(255,255,255,0.06)",
                    backgroundColor: "rgba(255,255,255,0.02)",
                  }}
                >
                  <p
                    className={`text-2xl font-light text-white ${jetbrains.className}`}
                  >
                    E2E
                  </p>
                  <p
                    className={`text-[10px] mt-1 uppercase tracking-wider ${jetbrains.className}`}
                    style={{ color: "rgba(255,255,255,0.25)" }}
                  >
                    Differentiable
                  </p>
                </div>
              </motion.div>
            </div>
          }
        />

        {/* ── Slide 7: Conclusion ──────────────────────────────────────── */}
        <Slide
          index={6}
          left={
            <div>
              <motion.p
                className={`text-xs tracking-[0.3em] uppercase mb-6 ${jetbrains.className}`}
                style={{ color: "rgba(168,85,247,0.5)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                06 &mdash; Conclusion
              </motion.p>

              <motion.h2
                className={`text-4xl lg:text-5xl font-light leading-tight mb-8 ${playfair.className}`}
                style={{ color: "rgba(255,255,255,0.9)" }}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1, duration: 0.7 }}
              >
                Quantum meets
                <br />
                astrobiology
              </motion.h2>

              <motion.p
                className="text-base leading-relaxed mb-5"
                style={{ color: "rgba(255,255,255,0.4)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.25, duration: 0.7 }}
              >
                ExoBiome demonstrates that quantum machine learning is not just
                a theoretical curiosity &mdash; it can deliver practical,
                state-of-the-art results on real scientific problems today.
              </motion.p>

              <motion.p
                className="text-base leading-relaxed mb-5"
                style={{ color: "rgba(255,255,255,0.4)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.35, duration: 0.7 }}
              >
                As quantum hardware scales and the Ariel telescope launches, this
                hybrid approach will become increasingly powerful for detecting
                the chemical signatures of habitable &mdash; and perhaps inhabited
                &mdash; worlds.
              </motion.p>

              <motion.div
                className="flex items-center gap-4 mt-10"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.5, duration: 0.6 }}
              >
                <div
                  className="w-10 h-px"
                  style={{ backgroundColor: "rgba(168,85,247,0.4)" }}
                />
                <p
                  className={`text-xs tracking-wider ${jetbrains.className}`}
                  style={{ color: "rgba(255,255,255,0.2)" }}
                >
                  HACK-4-SAGES 2026 &middot; Life Detection &amp; Biosignatures
                </p>
              </motion.div>
            </div>
          }
          right={
            <div className="space-y-5">
              <motion.p
                className={`text-xs tracking-[0.2em] uppercase mb-4 ${jetbrains.className}`}
                style={{ color: "rgba(255,255,255,0.2)" }}
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
              >
                Summary
              </motion.p>

              <div className="grid grid-cols-2 gap-4">
                <StatCard
                  label="mRMSE"
                  value="0.295"
                  sub="State-of-the-art"
                  delay={0.15}
                />
                <StatCard
                  label="vs ADC Winner"
                  value="-7.8%"
                  sub="Improvement"
                  delay={0.25}
                />
                <StatCard
                  label="Qubits"
                  value="12"
                  sub="Quantum circuit"
                  delay={0.35}
                />
                <StatCard
                  label="Parameters"
                  value="120K"
                  sub="Trainable"
                  delay={0.45}
                />
                <StatCard
                  label="Training"
                  value="~3 min"
                  sub="CPU only"
                  delay={0.55}
                />
                <StatCard
                  label="Molecules"
                  value="5"
                  sub="H₂O CO₂ CO CH₄ NH₃"
                  delay={0.65}
                />
              </div>

              <motion.div
                className="mt-6 p-5 rounded-xl border"
                style={{
                  borderColor: "rgba(168,85,247,0.15)",
                  background:
                    "linear-gradient(135deg, rgba(168,85,247,0.04) 0%, rgba(99,102,241,0.04) 100%)",
                }}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.75, duration: 0.6 }}
              >
                <p
                  className={`text-xs uppercase tracking-wider mb-3 ${jetbrains.className}`}
                  style={{ color: "rgba(168,85,247,0.5)" }}
                >
                  First of its kind
                </p>
                <p
                  className={`text-sm leading-relaxed ${playfair.className}`}
                  style={{ color: "rgba(255,255,255,0.5)" }}
                >
                  The first application of quantum machine learning to
                  biosignature detection from exoplanet transmission spectra.
                </p>
              </motion.div>
            </div>
          }
        />
      </div>
    </div>
  );
}
