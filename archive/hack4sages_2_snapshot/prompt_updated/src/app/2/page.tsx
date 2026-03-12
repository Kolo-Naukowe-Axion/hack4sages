"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import { motion, useScroll, useTransform, useInView } from "framer-motion";
import { Space_Grotesk, JetBrains_Mono } from "next/font/google";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jet",
});

// ─── Particle Field ────────────────────────────────────────────────────────────

function ParticleField({ activeSection }: { activeSection: number }) {
  const particles = useMemo(() => {
    const pts: {
      id: number;
      x: number;
      y: number;
      size: number;
      dur: number;
      delay: number;
      cluster: number;
      opacity: number;
      drift: number;
    }[] = [];
    const clusterCenters = [
      { x: 20, y: 10 },
      { x: 75, y: 25 },
      { x: 50, y: 50 },
      { x: 30, y: 75 },
      { x: 70, y: 85 },
      { x: 15, y: 45 },
      { x: 85, y: 60 },
      { x: 50, y: 15 },
    ];
    for (let i = 0; i < 300; i++) {
      const cluster = i % clusterCenters.length;
      const c = clusterCenters[cluster];
      const spread = 18;
      pts.push({
        id: i,
        x: c.x + (Math.random() - 0.5) * spread * 2,
        y: c.y + (Math.random() - 0.5) * spread * 2,
        size: Math.random() * 2.5 + 0.5,
        dur: Math.random() * 12 + 8,
        delay: Math.random() * -20,
        cluster,
        opacity: Math.random() * 0.4 + 0.1,
        drift: Math.random() * 6 + 2,
      });
    }
    return pts;
  }, []);

  return (
    <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
      {particles.map((p) => {
        const sectionCluster = activeSection % 8;
        const isActive =
          p.cluster === sectionCluster ||
          p.cluster === (sectionCluster + 1) % 8;
        return (
          <div
            key={p.id}
            className="absolute rounded-full"
            style={{
              left: `${p.x}%`,
              top: `${p.y}%`,
              width: p.size,
              height: p.size,
              backgroundColor: isActive
                ? `rgba(100, 210, 230, ${p.opacity * 2.5})`
                : `rgba(60, 130, 160, ${p.opacity})`,
              boxShadow: isActive
                ? `0 0 ${p.size * 4}px rgba(80, 200, 220, ${p.opacity * 1.5})`
                : "none",
              animation: `particleDrift${p.id % 4} ${p.dur}s ease-in-out infinite`,
              animationDelay: `${p.delay}s`,
              transition: "background-color 1.5s ease, box-shadow 1.5s ease",
            }}
          />
        );
      })}
    </div>
  );
}

// ─── Section Glow ──────────────────────────────────────────────────────────────

function SectionGlow({ color = "rgba(40, 140, 180, 0.08)" }: { color?: string }) {
  return (
    <div
      className="absolute inset-0 pointer-events-none z-0"
      style={{
        background: `radial-gradient(ellipse 60% 50% at 50% 50%, ${color}, transparent)`,
      }}
    />
  );
}

// ─── Grain Overlay ─────────────────────────────────────────────────────────────

function GrainOverlay() {
  return (
    <div
      className="fixed inset-0 z-50 pointer-events-none opacity-[0.035]"
      style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
        backgroundRepeat: "repeat",
        backgroundSize: "128px 128px",
      }}
    />
  );
}

// ─── Animated Section Wrapper ──────────────────────────────────────────────────

function RevealSection({
  children,
  id,
  onVisible,
  className = "",
}: {
  children: React.ReactNode;
  id: number;
  onVisible: (id: number) => void;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { amount: 0.5 });

  useEffect(() => {
    if (isInView) onVisible(id);
  }, [isInView, id, onVisible]);

  return (
    <section
      ref={ref}
      className={`relative min-h-screen w-full snap-start flex items-center justify-center ${className}`}
    >
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
        viewport={{ once: false, amount: 0.3 }}
        className="relative z-10 w-full max-w-5xl mx-auto px-6"
      >
        {children}
      </motion.div>
    </section>
  );
}

// ─── Architecture Node ─────────────────────────────────────────────────────────

function ArchNode({
  label,
  sub,
  glow,
  pulse,
  className = "",
}: {
  label: string;
  sub?: string;
  glow?: string;
  pulse?: boolean;
  className?: string;
}) {
  const border = glow || "rgba(80, 180, 210, 0.4)";
  const shadow = glow || "rgba(80, 180, 210, 0.15)";
  return (
    <div
      className={`relative flex flex-col items-center justify-center px-4 py-3 rounded-lg border text-center ${className}`}
      style={{
        borderColor: border,
        boxShadow: `0 0 20px ${shadow}`,
        background: "rgba(10, 15, 25, 0.8)",
        animation: pulse ? "nodePulse 3s ease-in-out infinite" : undefined,
      }}
    >
      <span
        className="text-xs font-mono tracking-wider uppercase"
        style={{ color: glow || "#64d2e6", fontFamily: "var(--font-jet)" }}
      >
        {label}
      </span>
      {sub && (
        <span
          className="text-[10px] mt-1 opacity-50"
          style={{ fontFamily: "var(--font-jet)" }}
        >
          {sub}
        </span>
      )}
    </div>
  );
}

function ConnectorLine({
  direction = "horizontal",
}: {
  direction?: "horizontal" | "vertical";
}) {
  if (direction === "vertical") {
    return (
      <div className="flex justify-center">
        <div
          className="w-px h-8"
          style={{
            background:
              "linear-gradient(to bottom, rgba(80,180,210,0.5), rgba(80,180,210,0.1))",
          }}
        />
      </div>
    );
  }
  return (
    <div className="flex items-center">
      <div
        className="h-px w-6"
        style={{
          background:
            "linear-gradient(to right, rgba(80,180,210,0.5), rgba(80,180,210,0.1))",
        }}
      />
    </div>
  );
}

// ─── Molecule Orb ──────────────────────────────────────────────────────────────

function MoleculeOrb({
  name,
  formula,
  score,
  color,
  delay,
}: {
  name: string;
  formula: string;
  score: number;
  color: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.5, y: 30 }}
      whileInView={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.7, delay, ease: "easeOut" }}
      viewport={{ once: false }}
      className="flex flex-col items-center"
    >
      <div
        className="relative w-28 h-28 rounded-full flex flex-col items-center justify-center"
        style={{
          background: `radial-gradient(circle at 40% 35%, ${color}22, ${color}08, transparent)`,
          border: `1px solid ${color}55`,
          boxShadow: `0 0 30px ${color}20, inset 0 0 20px ${color}10`,
        }}
      >
        <span
          className="text-lg font-bold"
          style={{ color, fontFamily: "var(--font-jet)" }}
        >
          {score.toFixed(3)}
        </span>
        <span
          className="text-[10px] mt-1 opacity-70 uppercase tracking-widest"
          style={{ fontFamily: "var(--font-jet)" }}
        >
          {formula}
        </span>
      </div>
      <span
        className="mt-3 text-xs opacity-40 uppercase tracking-widest"
        style={{ fontFamily: "var(--font-jet)" }}
      >
        {name}
      </span>
    </motion.div>
  );
}

// ─── Custom Tooltip ────────────────────────────────────────────────────────────

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { value: number; payload: { name: string } }[];
}) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div
      className="px-3 py-2 rounded border"
      style={{
        background: "rgba(5,10,20,0.95)",
        borderColor: "rgba(80,180,210,0.3)",
        fontFamily: "var(--font-jet)",
      }}
    >
      <p className="text-xs text-gray-400">{payload[0].payload.name}</p>
      <p className="text-sm" style={{ color: "#64d2e6" }}>
        mRMSE: {payload[0].value}
      </p>
    </div>
  );
}

// ─── Scroll Progress ──────────────────────────────────────────────────────────

function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const width = useTransform(scrollYProgress, [0, 1], ["0%", "100%"]);

  return (
    <div className="fixed top-0 left-0 right-0 h-px z-40">
      <motion.div
        className="h-full"
        style={{
          width,
          background:
            "linear-gradient(to right, transparent, rgba(80,200,220,0.6), rgba(80,200,220,0.2))",
        }}
      />
    </div>
  );
}

// ─── Nav Dots ──────────────────────────────────────────────────────────────────

function NavDots({
  count,
  active,
}: {
  count: number;
  active: number;
}) {
  return (
    <div className="fixed right-6 top-1/2 -translate-y-1/2 z-40 flex flex-col gap-3">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="w-1.5 h-1.5 rounded-full transition-all duration-500"
          style={{
            background:
              i === active
                ? "rgba(100, 210, 230, 0.9)"
                : "rgba(100, 210, 230, 0.15)",
            boxShadow:
              i === active ? "0 0 8px rgba(100, 210, 230, 0.5)" : "none",
            transform: i === active ? "scale(1.6)" : "scale(1)",
          }}
        />
      ))}
    </div>
  );
}

// ─── Main Page ─────────────────────────────────────────────────────────────────

const comparisonData = [
  { name: "Random Forest", mrmse: 1.2, color: "#3a4a5a" },
  { name: "CNN Baseline", mrmse: 0.85, color: "#4a5a6a" },
  { name: "ADC 2023 Winner", mrmse: 0.32, color: "#5a7a8a" },
  { name: "ExoBiome", mrmse: 0.295, color: "#64d2e6" },
];

const molecules = [
  { name: "Water", formula: "H₂O", score: 0.218, color: "#4fc3f7" },
  { name: "Carbon Dioxide", formula: "CO₂", score: 0.261, color: "#81c784" },
  { name: "Carbon Monoxide", formula: "CO", score: 0.327, color: "#ffb74d" },
  { name: "Methane", formula: "CH₄", score: 0.29, color: "#e57373" },
  { name: "Ammonia", formula: "NH₃", score: 0.378, color: "#ba68c8" },
];

export default function ExoBiomePage() {
  const [activeSection, setActiveSection] = useState(0);

  const handleVisible = (id: number) => setActiveSection(id);

  return (
    <div
      className={`${spaceGrotesk.variable} ${jetbrainsMono.variable} relative`}
      style={{ background: "#030308", color: "#e8eaed" }}
    >
      <style jsx global>{`
        @keyframes particleDrift0 {
          0%,
          100% {
            transform: translate(0, 0);
          }
          25% {
            transform: translate(4px, -6px);
          }
          50% {
            transform: translate(-3px, 5px);
          }
          75% {
            transform: translate(5px, 3px);
          }
        }
        @keyframes particleDrift1 {
          0%,
          100% {
            transform: translate(0, 0);
          }
          33% {
            transform: translate(-5px, -4px);
          }
          66% {
            transform: translate(6px, 7px);
          }
        }
        @keyframes particleDrift2 {
          0%,
          100% {
            transform: translate(0, 0);
          }
          20% {
            transform: translate(3px, 8px);
          }
          40% {
            transform: translate(-6px, 2px);
          }
          60% {
            transform: translate(2px, -5px);
          }
          80% {
            transform: translate(-4px, -3px);
          }
        }
        @keyframes particleDrift3 {
          0%,
          100% {
            transform: translate(0, 0);
          }
          50% {
            transform: translate(7px, -4px);
          }
        }
        @keyframes nodePulse {
          0%,
          100% {
            box-shadow: 0 0 20px rgba(140, 80, 220, 0.15);
          }
          50% {
            box-shadow: 0 0 35px rgba(140, 80, 220, 0.35),
              0 0 60px rgba(140, 80, 220, 0.1);
          }
        }
        @keyframes scoreBurst {
          0% {
            transform: scale(1);
            opacity: 0.3;
          }
          50% {
            transform: scale(1.6);
            opacity: 0;
          }
          100% {
            transform: scale(1);
            opacity: 0.3;
          }
        }
        .snap-container {
          scroll-snap-type: y mandatory;
          overflow-y: scroll;
          height: 100vh;
        }
        .snap-container::-webkit-scrollbar {
          display: none;
        }
      `}</style>

      <ParticleField activeSection={activeSection} />
      <GrainOverlay />
      <ScrollProgress />
      <NavDots count={7} active={activeSection} />

      <div className="snap-container">
        {/* ── SECTION 0: HERO ─────────────────────────────────────────── */}
        <RevealSection id={0} onVisible={handleVisible}>
          <SectionGlow color="rgba(40, 140, 180, 0.06)" />
          <div className="flex flex-col items-center text-center min-h-screen justify-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.2, delay: 0.2 }}
            >
              <p
                className="text-xs tracking-[0.4em] uppercase mb-6 opacity-40"
                style={{ fontFamily: "var(--font-jet)" }}
              >
                HACK-4-SAGES 2026 &middot; ETH Zurich
              </p>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.2, delay: 0.5 }}
              className="text-7xl md:text-8xl font-bold tracking-tight mb-4"
              style={{ fontFamily: "var(--font-space)" }}
            >
              <span style={{ color: "#64d2e6" }}>Exo</span>
              <span style={{ color: "#e8eaed" }}>Biome</span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.2, delay: 0.8 }}
              className="text-lg md:text-xl opacity-50 max-w-xl leading-relaxed"
              style={{ fontFamily: "var(--font-space)" }}
            >
              Quantum biosignature detection from exoplanet transmission spectra
            </motion.p>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.3 }}
              transition={{ duration: 1.5, delay: 1.5 }}
              className="mt-16 flex flex-col items-center"
            >
              <div
                className="w-px h-12"
                style={{
                  background:
                    "linear-gradient(to bottom, rgba(100,210,230,0.4), transparent)",
                }}
              />
              <span
                className="text-[10px] tracking-[0.3em] uppercase mt-3"
                style={{ fontFamily: "var(--font-jet)" }}
              >
                scroll
              </span>
            </motion.div>
          </div>
        </RevealSection>

        {/* ── SECTION 1: PROBLEM ─────────────────────────────────────── */}
        <RevealSection id={1} onVisible={handleVisible}>
          <SectionGlow color="rgba(60, 100, 180, 0.06)" />
          <div className="grid md:grid-cols-2 gap-16 items-center">
            <div>
              <p
                className="text-[10px] tracking-[0.4em] uppercase mb-4 opacity-30"
                style={{ fontFamily: "var(--font-jet)" }}
              >
                01 / The Challenge
              </p>
              <h2
                className="text-4xl md:text-5xl font-bold mb-6 leading-tight"
                style={{ fontFamily: "var(--font-space)" }}
              >
                Reading light from
                <br />
                <span style={{ color: "#64d2e6" }}>alien atmospheres</span>
              </h2>
              <p className="opacity-40 leading-relaxed text-sm max-w-md">
                When an exoplanet transits its star, starlight filters through
                its atmosphere. Each molecule absorbs at specific wavelengths,
                encoding its presence into the transmission spectrum. Retrieving
                molecular abundances from these faint signals is an inverse
                problem of extreme difficulty.
              </p>
            </div>
            <div className="flex flex-col gap-4">
              {[
                { label: "Spectrum Input", value: "52 wavelength bins", sub: "0.5 — 7.8 μm" },
                { label: "Aux Features", value: "Planet radius, mass, temperature", sub: "Stellar parameters" },
                { label: "Target Output", value: "log₁₀ VMR × 5 molecules", sub: "H₂O, CO₂, CO, CH₄, NH₃" },
              ].map((item, i) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0, x: 30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: i * 0.15 }}
                  viewport={{ once: false }}
                  className="p-4 rounded-lg border"
                  style={{
                    background: "rgba(10, 15, 25, 0.6)",
                    borderColor: "rgba(80, 180, 210, 0.15)",
                  }}
                >
                  <p
                    className="text-[10px] tracking-[0.3em] uppercase opacity-40 mb-1"
                    style={{ fontFamily: "var(--font-jet)" }}
                  >
                    {item.label}
                  </p>
                  <p
                    className="text-sm"
                    style={{ color: "#64d2e6", fontFamily: "var(--font-jet)" }}
                  >
                    {item.value}
                  </p>
                  <p className="text-[11px] opacity-30 mt-0.5">{item.sub}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </RevealSection>

        {/* ── SECTION 2: ARCHITECTURE ────────────────────────────────── */}
        <RevealSection id={2} onVisible={handleVisible}>
          <SectionGlow color="rgba(100, 60, 180, 0.06)" />
          <p
            className="text-[10px] tracking-[0.4em] uppercase mb-4 opacity-30 text-center"
            style={{ fontFamily: "var(--font-jet)" }}
          >
            02 / Architecture
          </p>
          <h2
            className="text-4xl md:text-5xl font-bold mb-12 text-center leading-tight"
            style={{ fontFamily: "var(--font-space)" }}
          >
            Quantum-classical
            <br />
            <span style={{ color: "#64d2e6" }}>hybrid pipeline</span>
          </h2>

          <div className="flex flex-col items-center gap-0">
            {/* Input row */}
            <div className="flex items-center gap-4 justify-center">
              <ArchNode label="Spectrum" sub="52 bins" />
              <ArchNode label="Aux Data" sub="planet params" />
            </div>
            <ConnectorLine direction="vertical" />

            {/* Encoder row */}
            <div className="flex items-center gap-4 justify-center">
              <ArchNode label="Spectral Encoder" sub="Conv1D + Dense" />
              <ConnectorLine />
              <ArchNode label="Aux Encoder" sub="Dense layers" />
            </div>
            <ConnectorLine direction="vertical" />

            {/* Fusion */}
            <ArchNode
              label="Fusion Layer"
              sub="Concatenate + Dense"
              className="w-56"
            />
            <ConnectorLine direction="vertical" />

            {/* Quantum */}
            <ArchNode
              label="Quantum Circuit"
              sub="12 qubits · variational"
              glow="rgba(140, 80, 220, 0.5)"
              pulse
              className="w-56"
            />
            <ConnectorLine direction="vertical" />

            {/* Output */}
            <div className="flex items-center gap-3 justify-center">
              {["H₂O", "CO₂", "CO", "CH₄", "NH₃"].map((mol) => (
                <div
                  key={mol}
                  className="px-3 py-2 rounded border text-center"
                  style={{
                    borderColor: "rgba(80, 180, 210, 0.25)",
                    background: "rgba(10, 15, 25, 0.7)",
                  }}
                >
                  <span
                    className="text-[10px]"
                    style={{
                      color: "#64d2e6",
                      fontFamily: "var(--font-jet)",
                    }}
                  >
                    {mol}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </RevealSection>

        {/* ── SECTION 3: RESULTS — HERO SCORE ────────────────────────── */}
        <RevealSection id={3} onVisible={handleVisible}>
          <SectionGlow color="rgba(80, 200, 220, 0.08)" />
          <div className="flex flex-col items-center text-center">
            <p
              className="text-[10px] tracking-[0.4em] uppercase mb-4 opacity-30"
              style={{ fontFamily: "var(--font-jet)" }}
            >
              03 / Performance
            </p>

            <div className="relative mb-8">
              {/* Burst rings */}
              <div
                className="absolute inset-0 rounded-full"
                style={{
                  background:
                    "radial-gradient(circle, rgba(100,210,230,0.15) 0%, transparent 70%)",
                  transform: "scale(3)",
                  animation: "scoreBurst 4s ease-in-out infinite",
                }}
              />
              <div
                className="absolute inset-0 rounded-full"
                style={{
                  background:
                    "radial-gradient(circle, rgba(100,210,230,0.08) 0%, transparent 60%)",
                  transform: "scale(5)",
                  animation: "scoreBurst 4s ease-in-out infinite 1s",
                }}
              />
              <motion.p
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                viewport={{ once: false }}
                className="relative text-8xl md:text-9xl font-bold"
                style={{
                  fontFamily: "var(--font-jet)",
                  color: "#64d2e6",
                  textShadow:
                    "0 0 40px rgba(100, 210, 230, 0.3), 0 0 80px rgba(100, 210, 230, 0.1)",
                }}
              >
                0.295
              </motion.p>
            </div>

            <p
              className="text-sm tracking-[0.2em] uppercase opacity-40 mb-2"
              style={{ fontFamily: "var(--font-jet)" }}
            >
              mean Relative RMSE
            </p>
            <p className="text-sm opacity-30 max-w-md">
              Outperforming the Ariel Data Challenge 2023 winning solution
              on 41,423 transmission spectra
            </p>
          </div>
        </RevealSection>

        {/* ── SECTION 4: COMPARISON CHART ─────────────────────────────── */}
        <RevealSection id={4} onVisible={handleVisible}>
          <SectionGlow color="rgba(40, 140, 180, 0.06)" />
          <p
            className="text-[10px] tracking-[0.4em] uppercase mb-4 opacity-30 text-center"
            style={{ fontFamily: "var(--font-jet)" }}
          >
            04 / Benchmarks
          </p>
          <h2
            className="text-3xl md:text-4xl font-bold mb-10 text-center"
            style={{ fontFamily: "var(--font-space)" }}
          >
            Against the <span style={{ color: "#64d2e6" }}>state of the art</span>
          </h2>

          <div className="w-full max-w-2xl mx-auto h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={comparisonData}
                layout="vertical"
                margin={{ left: 30, right: 30, top: 5, bottom: 5 }}
                barCategoryGap="30%"
              >
                <XAxis
                  type="number"
                  domain={[0, 1.4]}
                  tick={{ fill: "rgba(232,234,237,0.3)", fontSize: 11 }}
                  axisLine={{ stroke: "rgba(232,234,237,0.1)" }}
                  tickLine={false}
                  style={{ fontFamily: "var(--font-jet)" }}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: "rgba(232,234,237,0.5)", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={120}
                  style={{ fontFamily: "var(--font-jet)" }}
                />
                <Tooltip
                  content={<CustomTooltip />}
                  cursor={{ fill: "rgba(100,210,230,0.05)" }}
                />
                <Bar dataKey="mrmse" radius={[0, 4, 4, 0]}>
                  {comparisonData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.color}
                      style={{
                        filter:
                          entry.name === "ExoBiome"
                            ? "drop-shadow(0 0 8px rgba(100,210,230,0.5))"
                            : "none",
                      }}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="flex justify-center mt-6 gap-8">
            {comparisonData.map((d) => (
              <div key={d.name} className="text-center">
                <p
                  className="text-lg font-bold"
                  style={{
                    fontFamily: "var(--font-jet)",
                    color: d.color,
                  }}
                >
                  {d.mrmse}
                </p>
                <p className="text-[10px] opacity-30 mt-1 uppercase tracking-wider">
                  {d.name}
                </p>
              </div>
            ))}
          </div>
        </RevealSection>

        {/* ── SECTION 5: PER-MOLECULE ORBS ────────────────────────────── */}
        <RevealSection id={5} onVisible={handleVisible}>
          <SectionGlow color="rgba(60, 120, 180, 0.06)" />
          <p
            className="text-[10px] tracking-[0.4em] uppercase mb-4 opacity-30 text-center"
            style={{ fontFamily: "var(--font-jet)" }}
          >
            05 / Per-Molecule Breakdown
          </p>
          <h2
            className="text-3xl md:text-4xl font-bold mb-14 text-center"
            style={{ fontFamily: "var(--font-space)" }}
          >
            Five targets, <span style={{ color: "#64d2e6" }}>one model</span>
          </h2>

          <div className="flex justify-center items-end gap-8 md:gap-12 flex-wrap">
            {molecules.map((mol, i) => (
              <MoleculeOrb
                key={mol.formula}
                name={mol.name}
                formula={mol.formula}
                score={mol.score}
                color={mol.color}
                delay={i * 0.12}
              />
            ))}
          </div>

          <div className="mt-14 text-center">
            <p className="text-xs opacity-30 max-w-lg mx-auto">
              Volume mixing ratios predicted in log₁₀ scale. Lower mRMSE
              indicates higher retrieval accuracy. Each molecule&apos;s
              concentration spans orders of magnitude across the dataset.
            </p>
          </div>
        </RevealSection>

        {/* ── SECTION 6: HARDWARE & CLOSING ───────────────────────────── */}
        <RevealSection id={6} onVisible={handleVisible}>
          <SectionGlow color="rgba(140, 80, 220, 0.06)" />
          <div className="grid md:grid-cols-2 gap-16 items-center">
            <div>
              <p
                className="text-[10px] tracking-[0.4em] uppercase mb-4 opacity-30"
                style={{ fontFamily: "var(--font-jet)" }}
              >
                06 / Quantum Hardware
              </p>
              <h2
                className="text-3xl md:text-4xl font-bold mb-6 leading-tight"
                style={{ fontFamily: "var(--font-space)" }}
              >
                Executed on
                <br />
                <span style={{ color: "#b388ff" }}>real quantum processors</span>
              </h2>
              <p className="opacity-40 text-sm max-w-md leading-relaxed">
                Not a simulation. Our variational circuits ran on
                superconducting qubit hardware, proving that near-term quantum
                devices can contribute to scientific discovery.
              </p>
            </div>

            <div className="flex flex-col gap-5">
              {[
                {
                  name: "IQM Spark — Odra 5",
                  loc: "PWR Wroclaw, Poland",
                  detail: "5 superconducting qubits",
                  color: "#b388ff",
                },
                {
                  name: "VTT Q50",
                  loc: "VTT, Finland",
                  detail: "53 superconducting qubits",
                  color: "#64d2e6",
                },
              ].map((hw, i) => (
                <motion.div
                  key={hw.name}
                  initial={{ opacity: 0, x: 30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.6, delay: i * 0.2 }}
                  viewport={{ once: false }}
                  className="p-5 rounded-lg border"
                  style={{
                    background: "rgba(10, 15, 25, 0.6)",
                    borderColor: `${hw.color}30`,
                    boxShadow: `0 0 25px ${hw.color}08`,
                  }}
                >
                  <p
                    className="text-sm font-bold mb-1"
                    style={{
                      color: hw.color,
                      fontFamily: "var(--font-jet)",
                    }}
                  >
                    {hw.name}
                  </p>
                  <p className="text-xs opacity-40">{hw.loc}</p>
                  <p className="text-xs opacity-30 mt-1">{hw.detail}</p>
                </motion.div>
              ))}

              <div
                className="mt-4 p-4 rounded-lg border text-center"
                style={{
                  background: "rgba(10, 15, 25, 0.4)",
                  borderColor: "rgba(100, 210, 230, 0.1)",
                }}
              >
                <p
                  className="text-xs opacity-50 mb-2"
                  style={{ fontFamily: "var(--font-jet)" }}
                >
                  Dataset
                </p>
                <p
                  className="text-sm"
                  style={{
                    color: "#64d2e6",
                    fontFamily: "var(--font-jet)",
                  }}
                >
                  Ariel Data Challenge 2023
                </p>
                <p className="text-xs opacity-30 mt-1">
                  41,423 synthetic transmission spectra
                </p>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="mt-24 text-center pb-12">
            <div
              className="w-16 h-px mx-auto mb-6"
              style={{ background: "rgba(100, 210, 230, 0.2)" }}
            />
            <p
              className="text-xs tracking-[0.3em] uppercase opacity-20"
              style={{ fontFamily: "var(--font-jet)" }}
            >
              Life Detection &amp; Biosignatures
            </p>
            <p
              className="text-xs tracking-[0.2em] uppercase opacity-15 mt-2"
              style={{ fontFamily: "var(--font-jet)" }}
            >
              HACK-4-SAGES &middot; March 2026 &middot; ETH Zurich
            </p>
          </div>
        </RevealSection>
      </div>
    </div>
  );
}
