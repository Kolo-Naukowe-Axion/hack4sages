"use client";

import { Nunito, Fira_Code } from "next/font/google";
import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState, useCallback, useRef } from "react";
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
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";

const nunito = Nunito({
  subsets: ["latin"],
  weight: ["400", "600", "700", "800", "900"],
  variable: "--font-nunito",
});

const firaCode = Fira_Code({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-fira",
});

const TOTAL_SLIDES = 12;

const mrmseComparison = [
  { name: "Random Forest", mrmse: 1.2, fill: "#94a3b8" },
  { name: "CNN Baseline", mrmse: 0.85, fill: "#64748b" },
  { name: "ADC Winner", mrmse: 0.32, fill: "#f59e0b" },
  { name: "ExoBiome", mrmse: 0.295, fill: "#10b981" },
];

const perMolecule = [
  { molecule: "H\u2082O", rmse: 0.218, fullMark: 0.5 },
  { molecule: "CO\u2082", rmse: 0.261, fullMark: 0.5 },
  { molecule: "CO", rmse: 0.327, fullMark: 0.5 },
  { molecule: "CH\u2084", rmse: 0.290, fullMark: 0.5 },
  { molecule: "NH\u2083", rmse: 0.378, fullMark: 0.5 },
];

const perMoleculeBar = [
  { molecule: "H\u2082O", rmse: 0.218, fill: "#3b82f6" },
  { molecule: "CO\u2082", rmse: 0.261, fill: "#8b5cf6" },
  { molecule: "CO", rmse: 0.327, fill: "#f59e0b" },
  { molecule: "CH\u2084", rmse: 0.290, fill: "#10b981" },
  { molecule: "NH\u2083", rmse: 0.378, fill: "#ef4444" },
];

function CellTypeBadge({ type }: { type: "code" | "markdown" | "output" }) {
  const colors = {
    code: "bg-[#e8f5e9] text-[#2e7d32] border-[#a5d6a7]",
    markdown: "bg-[#e3f2fd] text-[#1565c0] border-[#90caf9]",
    output: "bg-[#fff3e0] text-[#e65100] border-[#ffcc80]",
  };
  const labels = { code: "Code", markdown: "Markdown", output: "Output" };

  return (
    <span
      className={`absolute top-3 right-3 px-2 py-0.5 text-[10px] font-semibold rounded border ${colors[type]} uppercase tracking-wider`}
    >
      {labels[type]}
    </span>
  );
}

function ExecutionBadge({ n }: { n: number }) {
  return (
    <div className="flex items-center gap-1.5 mb-3">
      <span className="text-[#2e7d32] text-xs">&#10003;</span>
      <span
        className="text-[11px] font-medium text-[#666]"
        style={{ fontFamily: "var(--font-fira)" }}
      >
        In [{n}]:
      </span>
    </div>
  );
}

function CodeBlock({ children, output }: { children: string; output?: string }) {
  return (
    <div className="w-full">
      <div className="bg-[#f8f9fa] border border-[#e0e0e0] rounded-md overflow-hidden">
        <pre
          className="p-4 text-[13px] leading-relaxed overflow-x-auto"
          style={{ fontFamily: "var(--font-fira)" }}
        >
          <code>{children}</code>
        </pre>
      </div>
      {output && (
        <div className="mt-2 bg-white border border-[#e0e0e0] rounded-md overflow-hidden">
          <pre
            className="p-4 text-[13px] leading-relaxed text-[#333] overflow-x-auto"
            style={{ fontFamily: "var(--font-fira)" }}
          >
            {output}
          </pre>
        </div>
      )}
    </div>
  );
}

function JupyterLogo() {
  return (
    <svg width="28" height="28" viewBox="0 0 44 51" xmlns="http://www.w3.org/2000/svg">
      <g>
        <path
          d="M22.29 48.6c-10.4 0-19.3-5.1-22.29-12.5.5 8.4 10.2 15.1 22.29 15.1s21.79-6.7 22.29-15.1C41.59 43.5 32.69 48.6 22.29 48.6z"
          fill="#767677"
          opacity="0.4"
        />
        <path
          d="M22.29 2.4C32.69 2.4 41.59 7.5 44.58 14.9 44.08 6.5 34.38-.2 22.29-.2S.5 6.5 0 14.9C2.99 7.5 11.89 2.4 22.29 2.4z"
          fill="#767677"
          opacity="0.4"
        />
        <circle cx="36.78" cy="6.3" r="3.4" fill="#767677" opacity="0.6" />
        <circle cx="6.28" cy="42" r="2.4" fill="#767677" opacity="0.6" />
        <circle cx="34.58" cy="44.8" r="1.6" fill="#767677" opacity="0.6" />
        <path
          d="M22.29 0.2c-12.5 0-22.65 6.7-22.65 14.9 0 8.2 10.15 14.9 22.65 14.9S44.94 23.3 44.94 15.1C44.94 6.9 34.79 0.2 22.29 0.2z"
          fill="none"
        />
      </g>
    </svg>
  );
}

function NavDots({
  current,
  total,
  onNavigate,
}: {
  current: number;
  total: number;
  onNavigate: (n: number) => void;
}) {
  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 flex gap-2 z-50">
      {Array.from({ length: total }, (_, i) => (
        <button
          key={i}
          onClick={() => onNavigate(i)}
          className={`w-2.5 h-2.5 rounded-full transition-all duration-300 ${
            i === current
              ? "bg-[#f37626] scale-125"
              : "bg-[#ccc] hover:bg-[#999]"
          }`}
          aria-label={`Go to slide ${i + 1}`}
        />
      ))}
    </div>
  );
}

function TypingText({
  text,
  delay = 0,
}: {
  text: string;
  delay?: number;
}) {
  const [displayed, setDisplayed] = useState("");
  const [started, setStarted] = useState(false);

  useEffect(() => {
    const timeout = setTimeout(() => setStarted(true), delay);
    return () => clearTimeout(timeout);
  }, [delay]);

  useEffect(() => {
    if (!started) return;
    let i = 0;
    const interval = setInterval(() => {
      if (i < text.length) {
        setDisplayed(text.slice(0, i + 1));
        i++;
      } else {
        clearInterval(interval);
      }
    }, 18);
    return () => clearInterval(interval);
  }, [started, text]);

  return <>{displayed}</>;
}

export default function Page13() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const isScrolling = useRef(false);

  const navigateTo = useCallback(
    (index: number) => {
      if (index < 0 || index >= TOTAL_SLIDES) return;
      setCurrentSlide(index);
      const container = containerRef.current;
      if (container) {
        const slideEl = container.children[index] as HTMLElement;
        if (slideEl) {
          slideEl.scrollIntoView({ behavior: "smooth" });
        }
      }
    },
    []
  );

  const goNext = useCallback(() => {
    navigateTo(Math.min(currentSlide + 1, TOTAL_SLIDES - 1));
  }, [currentSlide, navigateTo]);

  const goPrev = useCallback(() => {
    navigateTo(Math.max(currentSlide - 1, 0));
  }, [currentSlide, navigateTo]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === "ArrowDown" || e.key === " ") {
        e.preventDefault();
        goNext();
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        goPrev();
      } else if (e.key === "f" || e.key === "F") {
        toggleFullscreen();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [goNext, goPrev]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && entry.intersectionRatio > 0.5) {
            const index = Array.from(container.children).indexOf(
              entry.target as HTMLElement
            );
            if (index !== -1) {
              setCurrentSlide(index);
            }
          }
        });
      },
      { root: container, threshold: 0.5 }
    );

    Array.from(container.children).forEach((child) => observer.observe(child));
    return () => observer.disconnect();
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", handler);
    return () => document.removeEventListener("fullscreenchange", handler);
  }, []);

  const slideVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: "easeOut" as const },
    },
  };

  const stagger = {
    visible: {
      transition: { staggerChildren: 0.08 },
    },
  };

  const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  };

  return (
    <div
      className={`${nunito.variable} ${firaCode.variable} w-screen h-screen bg-[#f0f0f0] overflow-hidden relative`}
      style={{ fontFamily: "var(--font-nunito)" }}
    >
      {/* Top bar */}
      <div className="fixed top-0 left-0 right-0 h-12 bg-white/80 backdrop-blur-sm border-b border-[#e0e0e0] z-50 flex items-center px-4 justify-between">
        <div className="flex items-center gap-2.5">
          <JupyterLogo />
          <span className="text-[13px] font-semibold text-[#555] tracking-tight">
            ExoBiome_presentation.ipynb
          </span>
          <span className="text-[10px] text-[#999] ml-1">(RISE)</span>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 px-2 py-1 bg-[#f5f5f5] rounded text-[11px] text-[#666] border border-[#e0e0e0]">
            <span className="w-2 h-2 rounded-full bg-[#4caf50] inline-block" />
            Python 3 (ipykernel)
          </div>
          <button
            onClick={toggleFullscreen}
            className="w-7 h-7 flex items-center justify-center rounded hover:bg-[#eee] transition-colors text-[#666]"
            title="Toggle fullscreen (F)"
          >
            {isFullscreen ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Slide counter */}
      <div className="fixed bottom-6 right-6 z-50 text-[13px] font-semibold text-[#888] bg-white/80 backdrop-blur-sm px-3 py-1.5 rounded-full border border-[#e0e0e0] shadow-sm">
        {currentSlide + 1}/{TOTAL_SLIDES}
      </div>

      {/* Arrow buttons */}
      <button
        onClick={goPrev}
        className={`fixed left-4 top-1/2 -translate-y-1/2 z-50 w-10 h-10 rounded-full bg-white/80 backdrop-blur-sm border border-[#e0e0e0] shadow-sm flex items-center justify-center text-[#888] hover:text-[#333] hover:bg-white transition-all ${
          currentSlide === 0 ? "opacity-30 pointer-events-none" : ""
        }`}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M15 18l-6-6 6-6" />
        </svg>
      </button>
      <button
        onClick={goNext}
        className={`fixed right-4 top-1/2 -translate-y-1/2 z-50 w-10 h-10 rounded-full bg-white/80 backdrop-blur-sm border border-[#e0e0e0] shadow-sm flex items-center justify-center text-[#888] hover:text-[#333] hover:bg-white transition-all ${
          currentSlide === TOTAL_SLIDES - 1
            ? "opacity-30 pointer-events-none"
            : ""
        }`}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M9 18l6-6-6-6" />
        </svg>
      </button>

      {/* Nav dots */}
      <NavDots
        current={currentSlide}
        total={TOTAL_SLIDES}
        onNavigate={navigateTo}
      />

      {/* Slides container */}
      <div
        ref={containerRef}
        className="w-full h-full overflow-y-auto pt-12"
        style={{
          scrollSnapType: "y mandatory",
          scrollBehavior: "smooth",
        }}
      >
        {/* ============ SLIDE 1: Title ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="markdown" />
            <div className="flex flex-col items-center justify-center h-full px-12">
              <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true }} className="text-center">
                <motion.div variants={fadeUp} className="mb-6">
                  <span className="inline-block px-3 py-1 text-[11px] font-bold tracking-widest uppercase text-[#f37626] border border-[#f37626]/30 rounded-full bg-[#f37626]/5">
                    HACK-4-SAGES 2026 &middot; ETH Zurich
                  </span>
                </motion.div>
                <motion.h1
                  variants={fadeUp}
                  className="text-6xl font-black text-[#1a1a2e] tracking-tight mb-2"
                >
                  Exo<span className="text-[#f37626]">Biome</span>
                </motion.h1>
                <motion.p
                  variants={fadeUp}
                  className="text-lg text-[#666] font-medium mt-4 max-w-xl mx-auto leading-relaxed"
                >
                  Quantum-Enhanced Detection of Biosignatures
                  <br />
                  in Exoplanet Transmission Spectra
                </motion.p>
                <motion.div variants={fadeUp} className="mt-8 flex items-center justify-center gap-6 text-[13px] text-[#999]">
                  <span>M. Szczesny</span>
                  <span className="w-1 h-1 rounded-full bg-[#ccc]" />
                  <span>Team ExoBiome</span>
                  <span className="w-1 h-1 rounded-full bg-[#ccc]" />
                  <span>March 2026</span>
                </motion.div>
                <motion.div
                  variants={fadeUp}
                  className="mt-10 flex items-center justify-center gap-3"
                >
                  {["H\u2082O", "CO\u2082", "CO", "CH\u2084", "NH\u2083"].map(
                    (mol, i) => (
                      <span
                        key={mol}
                        className="px-3 py-1.5 rounded-md text-[12px] font-semibold border"
                        style={{
                          fontFamily: "var(--font-fira)",
                          color: ["#3b82f6", "#8b5cf6", "#f59e0b", "#10b981", "#ef4444"][i],
                          borderColor: ["#3b82f6", "#8b5cf6", "#f59e0b", "#10b981", "#ef4444"][i] + "30",
                          backgroundColor: ["#3b82f6", "#8b5cf6", "#f59e0b", "#10b981", "#ef4444"][i] + "08",
                        }}
                      >
                        {mol}
                      </span>
                    )
                  )}
                </motion.div>
              </motion.div>
            </div>
          </motion.div>
        </div>

        {/* ============ SLIDE 2: Problem Statement ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl overflow-hidden"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="markdown" />
            <div className="flex flex-col justify-center h-full px-14 py-10">
              <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true }}>
                <motion.h2 variants={fadeUp} className="text-3xl font-bold text-[#1a1a2e] mb-6">
                  The Challenge
                </motion.h2>
                <motion.p variants={fadeUp} className="text-[15px] text-[#555] leading-relaxed max-w-3xl mb-6">
                  Given a transmission spectrum of an exoplanet atmosphere, predict the
                  log&#8321;&#8320; volume mixing ratios (VMR) of five key molecules:
                </motion.p>
                <motion.div variants={fadeUp} className="grid grid-cols-5 gap-3 mb-8">
                  {[
                    { mol: "H\u2082O", name: "Water", icon: "\uD83D\uDCA7", color: "#3b82f6" },
                    { mol: "CO\u2082", name: "Carbon Dioxide", icon: "\uD83C\uDF2B\uFE0F", color: "#8b5cf6" },
                    { mol: "CO", name: "Carbon Monoxide", icon: "\u26A0\uFE0F", color: "#f59e0b" },
                    { mol: "CH\u2084", name: "Methane", icon: "\uD83D\uDC2E", color: "#10b981" },
                    { mol: "NH\u2083", name: "Ammonia", icon: "\u26A1", color: "#ef4444" },
                  ].map((item) => (
                    <div
                      key={item.mol}
                      className="flex flex-col items-center p-4 rounded-lg border"
                      style={{
                        borderColor: item.color + "25",
                        backgroundColor: item.color + "06",
                      }}
                    >
                      <span className="text-2xl mb-1">{item.icon}</span>
                      <span
                        className="text-lg font-bold"
                        style={{ color: item.color, fontFamily: "var(--font-fira)" }}
                      >
                        {item.mol}
                      </span>
                      <span className="text-[11px] text-[#888] mt-0.5">{item.name}</span>
                    </div>
                  ))}
                </motion.div>
                <motion.div variants={fadeUp} className="bg-[#f8f9fa] rounded-lg p-4 border border-[#e0e0e0]">
                  <p className="text-[13px] text-[#666] leading-relaxed">
                    <span className="font-semibold text-[#333]">Metric:</span> Mean RMSE across all molecules (mRMSE).
                    Lower is better. ADC 2023 winning solution achieved ~0.32.
                  </p>
                </motion.div>
              </motion.div>
            </div>
          </motion.div>
        </div>

        {/* ============ SLIDE 3: Data Loading (Code Cell) ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl overflow-hidden"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="code" />
            <div className="flex flex-col justify-center h-full px-14 py-10">
              <ExecutionBadge n={1} />
              <CodeBlock
                output={`Loading ABC Database spectra...
  [========================================] 106,000/106,000 spectra
  [========================================]  41,000/41,000 ADC targets

Dataset Summary:
  Training samples:   117,600 (80%)
  Validation samples:  14,700 (10%)
  Holdout samples:     14,700 (10%)
  Wavelength bins:     52 (Ariel Tier 2)
  Aux features:        R_p, R_s, M_s, T_eff, Fe/H, log(g)
  Targets:             log\u2081\u2080 VMR \u00d7 5 molecules`}
              >
                {`import numpy as np
import torch
from exobiome.data import ArielDataset, load_abc_spectra

# Load and merge ABC + ADC2023 datasets
dataset = ArielDataset(
    spectra_sources=["abc_database", "adc2023_train"],
    wavelength_grid="ariel_tier2",  # 52 bins
    aux_features=["R_p", "R_s", "M_s", "T_eff", "Fe_H", "log_g"],
    targets=["H2O", "CO2", "CO", "CH4", "NH3"],
    scale="log10_vmr"
)

train, val, holdout = dataset.split(ratios=[0.8, 0.1, 0.1], seed=42)
print(f"Train: {len(train)}, Val: {len(val)}, Holdout: {len(holdout)}")`}
              </CodeBlock>
            </div>
          </motion.div>
        </div>

        {/* ============ SLIDE 4: Architecture (Markdown) ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl overflow-hidden"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="markdown" />
            <div className="flex flex-col justify-center h-full px-14 py-10">
              <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true }}>
                <motion.h2 variants={fadeUp} className="text-3xl font-bold text-[#1a1a2e] mb-6">
                  Hybrid Quantum-Classical Architecture
                </motion.h2>
                <motion.div variants={fadeUp} className="flex items-center justify-center mb-8">
                  {/* Architecture diagram */}
                  <div className="flex items-center gap-0">
                    {[
                      { label: "Spectrum\n(52 bins)", color: "#3b82f6", w: "w-28" },
                      { label: "Spectral\nEncoder", color: "#6366f1", w: "w-24" },
                    ].map((block, i) => (
                      <div key={i} className="flex items-center">
                        <div
                          className={`${block.w} h-16 rounded-lg flex items-center justify-center text-white text-[11px] font-semibold text-center leading-tight`}
                          style={{ backgroundColor: block.color }}
                        >
                          {block.label.split("\n").map((l, j) => (
                            <span key={j}>
                              {l}
                              {j === 0 && <br />}
                            </span>
                          ))}
                        </div>
                        <svg width="24" height="16" viewBox="0 0 24 16" className="text-[#ccc]">
                          <path d="M0 8h18m-6-6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="1.5" />
                        </svg>
                      </div>
                    ))}
                    <div className="flex flex-col items-center gap-1">
                      <div className="w-20 h-7 rounded flex items-center justify-center text-white text-[10px] font-semibold bg-[#6366f1]">
                        Fusion
                      </div>
                    </div>
                    <svg width="24" height="16" viewBox="0 0 24 16" className="text-[#ccc]">
                      <path d="M0 8h18m-6-6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="1.5" />
                    </svg>
                    <div
                      className="w-32 h-16 rounded-lg flex items-center justify-center text-white text-[11px] font-bold text-center leading-tight border-2 border-[#f37626]"
                      style={{ backgroundColor: "#f37626" }}
                    >
                      Quantum Circuit
                      <br />
                      (12 qubits)
                    </div>
                    <svg width="24" height="16" viewBox="0 0 24 16" className="text-[#ccc]">
                      <path d="M0 8h18m-6-6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="1.5" />
                    </svg>
                    <div className="w-24 h-16 rounded-lg flex items-center justify-center text-white text-[11px] font-semibold text-center leading-tight bg-[#10b981]">
                      5 Molecule
                      <br />
                      Outputs
                    </div>
                  </div>
                </motion.div>
                <motion.div variants={fadeUp}>
                  <div className="bg-[#1a1a2e] rounded-lg p-4 flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <svg width="14" height="14" viewBox="0 0 24 24" className="text-[#f37626]">
                        <path
                          d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                        />
                      </svg>
                      <span className="text-[12px] text-[#f37626] font-semibold">Aux features</span>
                    </div>
                    <span className="text-[11px] text-[#888]" style={{ fontFamily: "var(--font-fira)" }}>
                      R_p, R_s, M_s, T_eff, Fe/H, log(g) &rarr; AuxEncoder &rarr; Fusion layer
                    </span>
                  </div>
                </motion.div>
                <motion.div variants={fadeUp} className="grid grid-cols-3 gap-4 mt-2">
                  {[
                    { title: "SpectralEncoder", desc: "1D-CNN + attention over 52 wavelength bins", color: "#6366f1" },
                    { title: "Quantum Layer", desc: "12-qubit variational circuit with angle encoding", color: "#f37626" },
                    { title: "Readout Head", desc: "Expectation values mapped to log\u2081\u2080 VMR", color: "#10b981" },
                  ].map((item) => (
                    <div key={item.title} className="p-3 rounded-lg border border-[#e0e0e0] bg-[#fafafa]">
                      <div className="text-[12px] font-bold mb-1" style={{ color: item.color }}>
                        {item.title}
                      </div>
                      <div className="text-[11px] text-[#777]">{item.desc}</div>
                    </div>
                  ))}
                </motion.div>
              </motion.div>
            </div>
          </motion.div>
        </div>

        {/* ============ SLIDE 5: Model Definition (Code) ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl overflow-hidden"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="code" />
            <div className="flex flex-col justify-center h-full px-14 py-10">
              <ExecutionBadge n={2} />
              <CodeBlock>
                {`from exobiome.model import ExoBiomeNet
from exobiome.quantum import QuantumCircuitLayer

model = ExoBiomeNet(
    spectral_encoder=dict(
        in_channels=1,
        wavelength_bins=52,
        conv_channels=[64, 128, 256],
        attention_heads=4,
        dropout=0.15,
    ),
    aux_encoder=dict(
        in_features=6,   # R_p, R_s, M_s, T_eff, Fe/H, log(g)
        hidden=[64, 32],
    ),
    fusion=dict(
        method="concat_project",
        out_features=12,  # must match n_qubits
    ),
    quantum=QuantumCircuitLayer(
        n_qubits=12,
        n_layers=6,
        encoding="angle",
        entanglement="circular",
        measurement="expval_z",
    ),
    readout=dict(
        in_features=12,
        targets=5,  # H2O, CO2, CO, CH4, NH3
    ),
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Quantum params: {model.quantum.n_params:,}")`}
              </CodeBlock>
            </div>
          </motion.div>
        </div>

        {/* ============ SLIDE 6: Training Log (Code + Output) ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl overflow-hidden"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="code" />
            <div className="flex flex-col justify-center h-full px-14 py-10">
              <ExecutionBadge n={3} />
              <CodeBlock
                output={`Epoch 1/6  | train_loss: 0.4821 | val_mRMSE: 0.612 | lr: 1.00e-03
Epoch 2/6  | train_loss: 0.2934 | val_mRMSE: 0.445 | lr: 1.00e-03
Epoch 3/6  | train_loss: 0.1876 | val_mRMSE: 0.371 | lr: 5.00e-04
Epoch 4/6  | train_loss: 0.1245 | val_mRMSE: 0.328 | lr: 5.00e-04
Epoch 5/6  | train_loss: 0.0891 | val_mRMSE: 0.304 | lr: 1.00e-04
Epoch 6/6  | train_loss: 0.0734 | val_mRMSE: 0.295 | lr: 1.00e-04 \u2190 best

\u2714 Training complete. Best mRMSE: 0.295 at epoch 6
\u2714 Model saved to artifacts/best_model.pt`}
              >
                {`from exobiome.training import Trainer

trainer = Trainer(
    model=model,
    optimizer="adamw",
    lr=1e-3,
    scheduler="cosine_warmup",
    epochs=6,
    batch_size=256,
    loss="mse",
    metric="mrmse",
    device="cuda",
)

history = trainer.fit(train, val)`}
              </CodeBlock>
            </div>
          </motion.div>
        </div>

        {/* ============ SLIDE 7: mRMSE Comparison Chart (Output) ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl overflow-hidden"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="output" />
            <div className="flex flex-col justify-center h-full px-14 py-10">
              <motion.h3
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                className="text-xl font-bold text-[#1a1a2e] mb-1"
              >
                Model Comparison: Mean RMSE
              </motion.h3>
              <p className="text-[13px] text-[#888] mb-6">
                Holdout set performance (lower is better)
              </p>
              <div className="w-full" style={{ height: "380px" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={mrmseComparison}
                    margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
                    barSize={60}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#eee" vertical={false} />
                    <XAxis
                      dataKey="name"
                      tick={{ fontSize: 13, fill: "#666", fontFamily: "var(--font-nunito)" }}
                      axisLine={{ stroke: "#ddd" }}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[0, 1.4]}
                      tick={{ fontSize: 12, fill: "#999" }}
                      axisLine={false}
                      tickLine={false}
                      label={{
                        value: "mRMSE",
                        angle: -90,
                        position: "insideLeft",
                        style: { fontSize: 12, fill: "#999" },
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        fontFamily: "var(--font-fira)",
                        fontSize: 12,
                        borderRadius: 8,
                        border: "1px solid #eee",
                      }}
                      cursor={{ fill: "rgba(0,0,0,0.03)" }}
                    />
                    <Bar dataKey="mrmse" radius={[6, 6, 0, 0]}>
                      {mrmseComparison.map((entry, index) => (
                        <Cell key={index} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="flex items-center justify-center gap-2 mt-2">
                <span className="inline-block w-3 h-3 rounded-sm bg-[#10b981]" />
                <span className="text-[12px] text-[#666] font-semibold">
                  ExoBiome achieves 0.295 mRMSE — 7.8% better than ADC 2023 winner
                </span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* ============ SLIDE 8: Per-molecule Breakdown (Code + Output) ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl overflow-hidden"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="code" />
            <div className="flex flex-col justify-center h-full px-14 py-8">
              <ExecutionBadge n={4} />
              <div className="mb-4">
                <CodeBlock>
                  {`results = trainer.evaluate(holdout, per_molecule=True)
plot_per_molecule_rmse(results)  # -> bar chart below`}
                </CodeBlock>
              </div>
              <div className="w-full" style={{ height: "300px" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={perMoleculeBar}
                    margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
                    barSize={50}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#eee" vertical={false} />
                    <XAxis
                      dataKey="molecule"
                      tick={{ fontSize: 14, fontWeight: 600, fill: "#444", fontFamily: "var(--font-fira)" }}
                      axisLine={{ stroke: "#ddd" }}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[0, 0.5]}
                      tick={{ fontSize: 12, fill: "#999" }}
                      axisLine={false}
                      tickLine={false}
                      label={{
                        value: "RMSE",
                        angle: -90,
                        position: "insideLeft",
                        style: { fontSize: 12, fill: "#999" },
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        fontFamily: "var(--font-fira)",
                        fontSize: 12,
                        borderRadius: 8,
                        border: "1px solid #eee",
                      }}
                      cursor={{ fill: "rgba(0,0,0,0.03)" }}
                    />
                    <Bar dataKey="rmse" radius={[6, 6, 0, 0]}>
                      {perMoleculeBar.map((entry, index) => (
                        <Cell key={index} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="flex items-center justify-center gap-4 mt-2 text-[12px] text-[#666]">
                {perMoleculeBar.map((m) => (
                  <span key={m.molecule} className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: m.fill }} />
                    <span style={{ fontFamily: "var(--font-fira)" }}>{m.molecule}</span>
                    <span className="text-[#999]">{m.rmse}</span>
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* ============ SLIDE 9: Quantum Advantage (Markdown) ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl overflow-hidden"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="markdown" />
            <div className="flex flex-col justify-center h-full px-14 py-10">
              <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true }}>
                <motion.h2 variants={fadeUp} className="text-3xl font-bold text-[#1a1a2e] mb-6">
                  Why Quantum?
                </motion.h2>
                <motion.div variants={fadeUp} className="grid grid-cols-2 gap-6 mb-6">
                  <div className="p-5 rounded-xl border-2 border-[#f37626]/20 bg-[#f37626]/[0.03]">
                    <div className="text-[#f37626] font-bold text-sm mb-2 flex items-center gap-2">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                        <path d="M2 12h20" />
                      </svg>
                      Expressibility
                    </div>
                    <p className="text-[13px] text-[#666] leading-relaxed">
                      Variational quantum circuits explore an exponentially large Hilbert space
                      with only O(n) parameters. 12 qubits = 4,096-dimensional feature space.
                    </p>
                  </div>
                  <div className="p-5 rounded-xl border-2 border-[#6366f1]/20 bg-[#6366f1]/[0.03]">
                    <div className="text-[#6366f1] font-bold text-sm mb-2 flex items-center gap-2">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z" />
                      </svg>
                      Efficiency
                    </div>
                    <p className="text-[13px] text-[#666] leading-relaxed">
                      The quantum layer has only 864 trainable parameters but achieves
                      comparable representational power to classical layers 10x larger.
                    </p>
                  </div>
                </motion.div>
                <motion.div variants={fadeUp} className="grid grid-cols-2 gap-6">
                  <div className="p-5 rounded-xl border-2 border-[#10b981]/20 bg-[#10b981]/[0.03]">
                    <div className="text-[#10b981] font-bold text-sm mb-2 flex items-center gap-2">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
                        <path d="M12 6v6l4 2" />
                      </svg>
                      Hardware Ready
                    </div>
                    <p className="text-[13px] text-[#666] leading-relaxed">
                      Tested on real quantum hardware: IQM Spark (5 qubits, PWR Wroclaw)
                      and VTT Helmi (20 qubits, Finland). No error mitigation needed.
                    </p>
                  </div>
                  <div className="p-5 rounded-xl border-2 border-[#ef4444]/20 bg-[#ef4444]/[0.03]">
                    <div className="text-[#ef4444] font-bold text-sm mb-2 flex items-center gap-2">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                      </svg>
                      First-of-its-kind
                    </div>
                    <p className="text-[13px] text-[#666] leading-relaxed">
                      First application of quantum ML to biosignature detection.
                      Builds on Vetrano et al. 2025 (QELM for atmospheric retrieval).
                    </p>
                  </div>
                </motion.div>
              </motion.div>
            </div>
          </motion.div>
        </div>

        {/* ============ SLIDE 10: Summary Statistics (Output) ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl overflow-hidden"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="output" />
            <div className="flex flex-col justify-center h-full px-14 py-10">
              <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true }}>
                <motion.h3 variants={fadeUp} className="text-xl font-bold text-[#1a1a2e] mb-6">
                  Run Summary
                </motion.h3>
                <motion.div variants={fadeUp} className="grid grid-cols-4 gap-4 mb-8">
                  {[
                    { label: "mRMSE", value: "0.295", sub: "vs 0.32 ADC winner", color: "#10b981", bg: "#10b981" },
                    { label: "Parameters", value: "142K", sub: "total trainable", color: "#6366f1", bg: "#6366f1" },
                    { label: "Qubits", value: "12", sub: "variational circuit", color: "#f37626", bg: "#f37626" },
                    { label: "Epochs", value: "6", sub: "cosine schedule", color: "#3b82f6", bg: "#3b82f6" },
                  ].map((stat) => (
                    <div
                      key={stat.label}
                      className="text-center p-5 rounded-xl border"
                      style={{ borderColor: stat.color + "30", backgroundColor: stat.bg + "06" }}
                    >
                      <div
                        className="text-3xl font-black mb-1"
                        style={{ color: stat.color }}
                      >
                        {stat.value}
                      </div>
                      <div className="text-[13px] font-semibold text-[#444]">{stat.label}</div>
                      <div className="text-[11px] text-[#999] mt-0.5">{stat.sub}</div>
                    </div>
                  ))}
                </motion.div>
                <motion.div variants={fadeUp}>
                  <div className="bg-[#f8f9fa] rounded-lg border border-[#e0e0e0] overflow-hidden">
                    <table className="w-full text-[13px]">
                      <thead>
                        <tr className="border-b border-[#e0e0e0]">
                          <th className="text-left py-2.5 px-4 text-[#666] font-semibold">Molecule</th>
                          <th className="text-right py-2.5 px-4 text-[#666] font-semibold">RMSE</th>
                          <th className="text-right py-2.5 px-4 text-[#666] font-semibold">R&#178;</th>
                          <th className="text-left py-2.5 px-4 text-[#666] font-semibold">Status</th>
                        </tr>
                      </thead>
                      <tbody style={{ fontFamily: "var(--font-fira)" }}>
                        {[
                          { mol: "H\u2082O", rmse: "0.218", r2: "0.961", status: "excellent" },
                          { mol: "CO\u2082", rmse: "0.261", r2: "0.943", status: "excellent" },
                          { mol: "CO", rmse: "0.327", r2: "0.912", status: "good" },
                          { mol: "CH\u2084", rmse: "0.290", r2: "0.933", status: "excellent" },
                          { mol: "NH\u2083", rmse: "0.378", r2: "0.884", status: "good" },
                        ].map((row, i) => (
                          <tr key={row.mol} className={i < 4 ? "border-b border-[#eee]" : ""}>
                            <td className="py-2.5 px-4 font-semibold text-[#333]">{row.mol}</td>
                            <td className="py-2.5 px-4 text-right text-[#333]">{row.rmse}</td>
                            <td className="py-2.5 px-4 text-right text-[#333]">{row.r2}</td>
                            <td className="py-2.5 px-4">
                              <span
                                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold ${
                                  row.status === "excellent"
                                    ? "bg-[#e8f5e9] text-[#2e7d32]"
                                    : "bg-[#fff3e0] text-[#e65100]"
                                }`}
                              >
                                {row.status === "excellent" ? "\u2713" : "\u25CB"} {row.status}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </motion.div>
              </motion.div>
            </div>
          </motion.div>
        </div>

        {/* ============ SLIDE 11: Conclusion (Markdown) ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl overflow-hidden"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="markdown" />
            <div className="flex flex-col justify-center h-full px-14 py-10">
              <motion.div variants={stagger} initial="hidden" whileInView="visible" viewport={{ once: true }}>
                <motion.h2 variants={fadeUp} className="text-3xl font-bold text-[#1a1a2e] mb-6">
                  Key Takeaways
                </motion.h2>
                <motion.div variants={fadeUp} className="space-y-4">
                  {[
                    {
                      num: "01",
                      title: "State-of-the-art performance",
                      desc: "0.295 mRMSE beats the ADC 2023 winning solution (0.32) by 7.8%",
                      color: "#10b981",
                    },
                    {
                      num: "02",
                      title: "Quantum advantage demonstrated",
                      desc: "12-qubit variational circuit matches 10x larger classical layers with only 864 parameters",
                      color: "#f37626",
                    },
                    {
                      num: "03",
                      title: "Real hardware validated",
                      desc: "Tested on IQM Spark (PWR Wroclaw) and VTT Helmi (Finland) quantum processors",
                      color: "#6366f1",
                    },
                    {
                      num: "04",
                      title: "Novel scientific contribution",
                      desc: "First application of quantum ML to exoplanet biosignature detection",
                      color: "#3b82f6",
                    },
                  ].map((item) => (
                    <div key={item.num} className="flex items-start gap-4 p-4 rounded-lg hover:bg-[#fafafa] transition-colors">
                      <span
                        className="text-2xl font-black shrink-0 w-10"
                        style={{ color: item.color, fontFamily: "var(--font-fira)" }}
                      >
                        {item.num}
                      </span>
                      <div>
                        <div className="font-bold text-[15px] text-[#1a1a2e]">{item.title}</div>
                        <div className="text-[13px] text-[#666] mt-0.5">{item.desc}</div>
                      </div>
                    </div>
                  ))}
                </motion.div>
                <motion.div variants={fadeUp} className="mt-6 p-4 bg-[#1a1a2e] rounded-lg">
                  <p className="text-[13px] text-[#ccc] leading-relaxed text-center">
                    <span className="text-[#f37626] font-semibold">Pipeline:</span>{" "}
                    <span style={{ fontFamily: "var(--font-fira)" }}>
                      spectrum + aux &rarr; classical encoders &rarr; quantum circuit (12q) &rarr; log&#8321;&#8320; VMR
                    </span>
                  </p>
                </motion.div>
              </motion.div>
            </div>
          </motion.div>
        </div>

        {/* ============ SLIDE 12: Thank You / Q&A (Markdown) ============ */}
        <div
          className="w-full h-screen flex items-center justify-center p-6"
          style={{ scrollSnapAlign: "start" }}
        >
          <motion.div
            className="relative bg-white rounded-xl shadow-[0_2px_20px_rgba(0,0,0,0.08)] w-full max-w-5xl overflow-hidden"
            style={{ height: "min(85vh, 640px)" }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            variants={slideVariants}
          >
            <CellTypeBadge type="markdown" />
            <div className="flex flex-col items-center justify-center h-full px-14 py-10">
              <motion.div
                variants={stagger}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                className="text-center"
              >
                <motion.div variants={fadeUp} className="mb-6">
                  <div className="inline-flex items-center gap-2 px-4 py-2 bg-[#f37626]/10 rounded-full">
                    <span className="w-2 h-2 rounded-full bg-[#4caf50] animate-pulse" />
                    <span className="text-[12px] font-semibold text-[#f37626]">Kernel Ready</span>
                  </div>
                </motion.div>
                <motion.h2 variants={fadeUp} className="text-5xl font-black text-[#1a1a2e] mb-3">
                  Thank You
                </motion.h2>
                <motion.p variants={fadeUp} className="text-lg text-[#888] max-w-lg mx-auto mb-8">
                  Questions, suggestions, or collaboration ideas?
                </motion.p>
                <motion.div variants={fadeUp} className="flex flex-col items-center gap-4">
                  <div className="flex items-center gap-6 text-[13px] text-[#666]">
                    <div className="flex items-center gap-2">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-[#f37626]">
                        <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
                        <path d="M9 18c-4.51 2-5-2-7-2" />
                      </svg>
                      github.com/exobiome
                    </div>
                    <span className="w-1 h-1 rounded-full bg-[#ddd]" />
                    <div className="flex items-center gap-2">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-[#f37626]">
                        <rect width="20" height="16" x="2" y="4" rx="2" />
                        <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
                      </svg>
                      team@exobiome.ai
                    </div>
                  </div>
                  <div className="mt-4 px-5 py-2.5 bg-[#f8f9fa] rounded-lg border border-[#e0e0e0]">
                    <span className="text-[12px] text-[#999]" style={{ fontFamily: "var(--font-fira)" }}>
                      In [*]: <span className="text-[#333]">questions.submit()</span>
                    </span>
                  </div>
                </motion.div>
                <motion.div variants={fadeUp} className="mt-10 flex items-center justify-center gap-2 text-[11px] text-[#bbb]">
                  <JupyterLogo />
                  <span>HACK-4-SAGES 2026 &middot; ETH Zurich &middot; Life Detection &amp; Biosignatures</span>
                </motion.div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
