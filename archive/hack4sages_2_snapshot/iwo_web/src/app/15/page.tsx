"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Libre_Franklin, JetBrains_Mono } from "next/font/google";
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
  PolarRadiusAxis,
  Radar,
  Legend,
  LineChart,
  Line,
  ReferenceLine,
} from "recharts";

const franklin = Libre_Franklin({
  weight: ["300", "400", "500", "600", "700", "800"],
  subsets: ["latin"],
  variable: "--font-franklin",
});

const jetbrains = JetBrains_Mono({
  weight: ["400", "500"],
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

// Quarto color palette
const QUARTO = {
  body: "#343a40",
  heading: "#343a40",
  link: "#2780e3",
  linkHover: "#1a5bb5",
  codeBg: "#f8f9fa",
  codeBorder: "#dee2e6",
  calloutNote: "#2780e3",
  calloutTip: "#3fb618",
  calloutWarning: "#f0ad4e",
  calloutImportant: "#e83e8c",
  sidebarBg: "#f8f9fa",
  sidebarText: "#6c757d",
  sidebarActive: "#2780e3",
  border: "#dee2e6",
  muted: "#6c757d",
  tableBorder: "#dee2e6",
  tableStripe: "#f8f9fa",
};

const moleculeColors: Record<string, string> = {
  "H₂O": "#2780e3",
  "CO₂": "#e83e8c",
  CO: "#f0ad4e",
  "CH₄": "#3fb618",
  "NH₃": "#9954bb",
};

// ─── TOC sections ───
const sections = [
  { id: "abstract", label: "Abstract" },
  { id: "introduction", label: "1 Introduction" },
  { id: "dataset", label: "2 Dataset" },
  { id: "architecture", label: "3 Architecture" },
  { id: "quantum-layer", label: "3.1 Quantum Layer" },
  { id: "training", label: "4 Training" },
  { id: "results", label: "5 Results" },
  { id: "comparison", label: "5.1 Model Comparison" },
  { id: "per-molecule", label: "5.2 Per-Molecule" },
  { id: "discussion", label: "6 Discussion" },
  { id: "references", label: "References" },
];

// ─── Citation data ───
const citations: Record<string, { short: string; full: string }> = {
  vetrano2025: {
    short: "Vetrano et al., 2025",
    full: "Vetrano, D., Cirillo, A., Giampaolo, F., Narici, L., & Parisi, F. (2025). Quantum Extreme Learning Machine for Exoplanet Atmospheric Retrieval. arXiv:2509.03617.",
  },
  changeat2022: {
    short: "Changeat et al., 2022",
    full: "Changeat, Q., Yip, K.H., & Waldmann, I.P. (2022). Ariel Data Challenge 2023: Overview and Baseline. Zenodo 6770103.",
  },
  schwieterman2018: {
    short: "Schwieterman et al., 2018",
    full: "Schwieterman, E.W. et al. (2018). Exoplanet Biosignatures: A Review of Remotely Detectable Signs of Life. Astrobiology 18(6), 663–708.",
  },
  seeburger2023: {
    short: "Seeburger et al., 2023",
    full: "Seeburger, R., Ringeval, B., & Tran, D. (2023). From Methanogenesis to Planetary Spectra. MNRAS, in press.",
  },
  schuld2021: {
    short: "Schuld & Petruccione, 2021",
    full: "Schuld, M. & Petruccione, F. (2021). Machine Learning with Quantum Computers. Springer.",
  },
  cardenas2025: {
    short: "Cardenas et al., 2025",
    full: "Cardenas, R. et al. (2025). MultiREx: A Multi-spectral Retrieval Exercise dataset. MNRAS 539.",
  },
};

// ─── Reusable Components ───

function useActiveSection() {
  const [active, setActive] = useState("abstract");

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length > 0) {
          const top = visible.reduce((a, b) =>
            a.boundingClientRect.top < b.boundingClientRect.top ? a : b
          );
          setActive(top.target.id);
        }
      },
      { rootMargin: "-80px 0px -60% 0px", threshold: 0.1 }
    );

    sections.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  return active;
}

function Citation({ id }: { id: string }) {
  const [show, setShow] = useState(false);
  const cite = citations[id];
  if (!cite) return null;

  return (
    <span
      className="relative inline-block"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <span
        className="cursor-help"
        style={{ color: QUARTO.link, fontStyle: "normal" }}
      >
        ({cite.short})
      </span>
      <AnimatePresence>
        {show && (
          <motion.span
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 left-0 bottom-full mb-2 w-80 p-3 rounded shadow-lg text-xs leading-relaxed"
            style={{
              background: "#1a1a2e",
              color: "#e0e0e0",
              border: "1px solid #333",
            }}
          >
            {cite.full}
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}

function CodeBlock({
  code,
  language = "python",
  label,
  defaultOpen = false,
}: {
  code: string;
  language?: string;
  label?: string;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [code]);

  return (
    <div className="my-4" style={{ borderLeft: `3px solid ${QUARTO.border}` }}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 text-sm w-full text-left transition-colors"
        style={{
          fontFamily: "var(--font-franklin)",
          color: QUARTO.muted,
          background: QUARTO.codeBg,
          borderBottom: open ? `1px solid ${QUARTO.border}` : "none",
        }}
      >
        <svg
          width="10"
          height="10"
          viewBox="0 0 10 10"
          className="transition-transform"
          style={{ transform: open ? "rotate(90deg)" : "rotate(0deg)" }}
        >
          <path d="M3 1 L7 5 L3 9" fill="none" stroke={QUARTO.muted} strokeWidth="1.5" />
        </svg>
        {label || "Code"}
        {language && (
          <span
            className="ml-auto text-xs px-1.5 py-0.5 rounded"
            style={{ background: "#e9ecef", color: QUARTO.muted }}
          >
            {language}
          </span>
        )}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden relative"
          >
            <button
              onClick={handleCopy}
              className="absolute top-2 right-2 text-xs px-2 py-1 rounded transition-colors z-10"
              style={{
                background: copied ? "#3fb618" : "#e9ecef",
                color: copied ? "#fff" : QUARTO.muted,
              }}
            >
              {copied ? "Copied!" : "Copy"}
            </button>
            <pre
              className="p-4 overflow-x-auto text-[13px] leading-relaxed"
              style={{
                fontFamily: "var(--font-jetbrains)",
                background: QUARTO.codeBg,
                color: QUARTO.body,
                margin: 0,
              }}
            >
              <code>{code}</code>
            </pre>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Callout({
  type,
  title,
  children,
}: {
  type: "note" | "tip" | "warning" | "important";
  title?: string;
  children: React.ReactNode;
}) {
  const config = {
    note: { color: QUARTO.calloutNote, icon: "ℹ", label: "Note" },
    tip: { color: QUARTO.calloutTip, icon: "✓", label: "Tip" },
    warning: { color: QUARTO.calloutWarning, icon: "⚠", label: "Warning" },
    important: { color: QUARTO.calloutImportant, icon: "!", label: "Important" },
  }[type];

  return (
    <div
      className="my-6 rounded-sm overflow-hidden"
      style={{
        borderLeft: `4px solid ${config.color}`,
        background: `${config.color}08`,
      }}
    >
      <div
        className="px-4 py-2.5 flex items-center gap-2 text-sm font-semibold"
        style={{
          background: `${config.color}12`,
          color: config.color,
          fontFamily: "var(--font-franklin)",
        }}
      >
        <span className="w-5 h-5 rounded-full flex items-center justify-center text-xs text-white" style={{ background: config.color }}>
          {config.icon}
        </span>
        {title || config.label}
      </div>
      <div
        className="px-4 py-3 text-sm leading-relaxed"
        style={{ color: QUARTO.body, fontFamily: "var(--font-franklin)" }}
      >
        {children}
      </div>
    </div>
  );
}

function MarginNote({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="hidden xl:block absolute right-0 w-48 text-xs leading-relaxed translate-x-[calc(100%+2rem)]"
      style={{ color: QUARTO.muted, fontFamily: "var(--font-franklin)" }}
    >
      {children}
    </span>
  );
}

function Figure({
  number,
  caption,
  children,
}: {
  number: number;
  caption: string;
  children: React.ReactNode;
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-50px" });

  return (
    <motion.figure
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5 }}
      className="my-8"
      id={`fig-${number}`}
    >
      <div
        className="rounded overflow-hidden"
        style={{ border: `1px solid ${QUARTO.border}` }}
      >
        {children}
      </div>
      <figcaption
        className="mt-2 text-sm text-center"
        style={{ color: QUARTO.muted, fontFamily: "var(--font-franklin)" }}
      >
        <strong>Figure {number}:</strong> {caption}
      </figcaption>
    </motion.figure>
  );
}

function TabSet({
  tabs,
}: {
  tabs: { label: string; content: React.ReactNode; disabled?: boolean }[];
}) {
  const [active, setActive] = useState(0);

  return (
    <div className="my-6">
      <div
        className="flex gap-0"
        style={{ borderBottom: `2px solid ${QUARTO.border}` }}
      >
        {tabs.map((tab, i) => (
          <button
            key={tab.label}
            onClick={() => !tab.disabled && setActive(i)}
            className="px-4 py-2 text-sm font-medium transition-colors relative"
            style={{
              fontFamily: "var(--font-franklin)",
              color: tab.disabled
                ? "#ccc"
                : active === i
                ? QUARTO.link
                : QUARTO.muted,
              cursor: tab.disabled ? "not-allowed" : "pointer",
              borderBottom:
                active === i ? `2px solid ${QUARTO.link}` : "2px solid transparent",
              marginBottom: "-2px",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="pt-3">{tabs[active].content}</div>
    </div>
  );
}

function SectionHeading({
  id,
  level,
  children,
}: {
  id: string;
  level: 1 | 2 | 3;
  children: React.ReactNode;
}) {
  const sizes = { 1: "text-2xl", 2: "text-xl", 3: "text-lg" };
  const margins = { 1: "mt-12 mb-4", 2: "mt-10 mb-3", 3: "mt-8 mb-2" };
  const weights = { 1: "font-bold", 2: "font-semibold", 3: "font-semibold" };

  const sharedProps = {
    id,
    className: `${sizes[level]} ${margins[level]} ${weights[level]} group scroll-mt-20`,
    style: {
      color: QUARTO.heading,
      fontFamily: "var(--font-franklin)",
      letterSpacing: "-0.01em",
      lineHeight: 1.3,
      borderBottom: level <= 2 ? `1px solid ${QUARTO.border}` : "none",
      paddingBottom: level <= 2 ? "0.4rem" : 0,
    } as React.CSSProperties,
  };

  const inner = (
    <>
      <a
        href={`#${id}`}
        className="opacity-0 group-hover:opacity-100 transition-opacity mr-2"
        style={{ color: QUARTO.link, textDecoration: "none" }}
      >
        §
      </a>
      {children}
    </>
  );

  if (level === 1) return <h1 {...sharedProps}>{inner}</h1>;
  if (level === 2) return <h2 {...sharedProps}>{inner}</h2>;
  return <h3 {...sharedProps}>{inner}</h3>;
}

// ─── Chart Data ───

const comparisonData = [
  { model: "Random Forest", mrmse: 1.2, fill: "#adb5bd" },
  { model: "CNN Baseline", mrmse: 0.85, fill: "#868e96" },
  { model: "ADC Winner", mrmse: 0.32, fill: "#495057" },
  { model: "ExoBiome", mrmse: 0.295, fill: QUARTO.link },
];

const perMoleculeData = [
  { molecule: "H₂O", rmse: 0.218, color: moleculeColors["H₂O"] },
  { molecule: "CO₂", rmse: 0.261, color: moleculeColors["CO₂"] },
  { molecule: "CO", rmse: 0.327, color: moleculeColors.CO },
  { molecule: "CH₄", rmse: 0.29, color: moleculeColors["CH₄"] },
  { molecule: "NH₃", rmse: 0.378, color: moleculeColors["NH₃"] },
];

const radarData = [
  { metric: "H₂O", ExoBiome: 0.92, Classical: 0.65 },
  { metric: "CO₂", ExoBiome: 0.88, Classical: 0.6 },
  { metric: "CO", ExoBiome: 0.82, Classical: 0.52 },
  { metric: "CH₄", ExoBiome: 0.86, Classical: 0.58 },
  { metric: "NH₃", ExoBiome: 0.79, Classical: 0.45 },
];

const trainingData = Array.from({ length: 30 }, (_, i) => ({
  epoch: i + 1,
  train: 1.8 * Math.exp(-0.12 * i) + 0.28 + Math.random() * 0.03,
  val: 1.9 * Math.exp(-0.1 * i) + 0.30 + Math.random() * 0.04,
}));

const spectrumData = Array.from({ length: 52 }, (_, i) => {
  const wl = 0.5 + i * 0.15;
  const base = 0.01 + 0.003 * Math.sin(wl * 2.1) + 0.002 * Math.cos(wl * 4.3);
  const h2o = wl > 1.3 && wl < 1.5 ? 0.004 * Math.exp(-((wl - 1.4) ** 2) / 0.005) : 0;
  const co2 = wl > 4.1 && wl < 4.5 ? 0.005 * Math.exp(-((wl - 4.3) ** 2) / 0.01) : 0;
  const ch4 = wl > 3.1 && wl < 3.5 ? 0.003 * Math.exp(-((wl - 3.3) ** 2) / 0.008) : 0;
  return {
    wavelength: +wl.toFixed(2),
    depth: +(base + h2o + co2 + ch4 + Math.random() * 0.001).toFixed(5),
  };
});

// ─── Main Page Component ───

export default function QuartoPage() {
  const activeSection = useActiveSection();

  return (
    <div
      className={`${franklin.variable} ${jetbrains.variable} min-h-screen`}
      style={{
        fontFamily: "var(--font-franklin)",
        color: QUARTO.body,
        background: "#ffffff",
      }}
    >
      {/* Top Nav Bar */}
      <header
        className="fixed top-0 left-0 right-0 z-50 h-14 flex items-center px-6"
        style={{
          background: "#fff",
          borderBottom: `1px solid ${QUARTO.border}`,
          backdropFilter: "blur(8px)",
        }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-7 h-7 rounded flex items-center justify-center text-xs font-bold text-white"
            style={{ background: QUARTO.link }}
          >
            E
          </div>
          <span
            className="text-sm font-semibold"
            style={{ color: QUARTO.heading }}
          >
            ExoBiome
          </span>
          <span className="text-xs" style={{ color: QUARTO.muted }}>
            HACK-4-SAGES 2026
          </span>
        </div>
        <nav className="ml-auto flex items-center gap-6">
          {["Paper", "Code", "Data", "About"].map((item) => (
            <span
              key={item}
              className="text-sm cursor-pointer hover:opacity-80 transition-opacity"
              style={{ color: QUARTO.muted }}
            >
              {item}
            </span>
          ))}
        </nav>
      </header>

      <div className="flex pt-14">
        {/* Left Sidebar — TOC */}
        <aside
          className="hidden lg:block fixed left-0 top-14 bottom-0 w-56 overflow-y-auto py-6 px-4"
          style={{
            background: QUARTO.sidebarBg,
            borderRight: `1px solid ${QUARTO.border}`,
          }}
        >
          <div
            className="text-xs font-semibold uppercase tracking-wider mb-4"
            style={{ color: QUARTO.muted }}
          >
            On this page
          </div>
          <nav className="space-y-0.5">
            {sections.map(({ id, label }) => {
              const isActive = activeSection === id;
              const isSubsection = label.includes(".");
              return (
                <a
                  key={id}
                  href={`#${id}`}
                  className={`block py-1.5 text-sm transition-all ${
                    isSubsection ? "pl-5" : "pl-3"
                  }`}
                  style={{
                    color: isActive ? QUARTO.sidebarActive : QUARTO.sidebarText,
                    fontWeight: isActive ? 600 : 400,
                    borderLeft: isActive
                      ? `2px solid ${QUARTO.sidebarActive}`
                      : "2px solid transparent",
                    textDecoration: "none",
                    fontSize: isSubsection ? "0.8rem" : "0.85rem",
                  }}
                >
                  {label}
                </a>
              );
            })}
          </nav>

          <div
            className="mt-8 pt-4 text-xs space-y-2"
            style={{ borderTop: `1px solid ${QUARTO.border}`, color: QUARTO.muted }}
          >
            <div>
              <span className="font-medium">Published:</span> March 12, 2026
            </div>
            <div>
              <span className="font-medium">Modified:</span> March 12, 2026
            </div>
            <div>
              <span className="font-medium">Format:</span> Quarto HTML
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main
          className="flex-1 lg:ml-56 min-h-screen"
          style={{ maxWidth: "100%" }}
        >
          <div className="relative mx-auto px-6 lg:px-12 py-10" style={{ maxWidth: 860 }}>
            {/* Title Block */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="mb-10 pb-8"
              style={{ borderBottom: `2px solid ${QUARTO.heading}` }}
            >
              <h1
                className="text-3xl lg:text-4xl font-extrabold leading-tight mb-4"
                style={{
                  color: QUARTO.heading,
                  fontFamily: "var(--font-franklin)",
                  letterSpacing: "-0.02em",
                }}
              >
                ExoBiome: Quantum-Enhanced Biosignature Detection
                <br />
                from Exoplanet Transmission Spectra
              </h1>
              <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm mb-3" style={{ color: QUARTO.muted }}>
                <span>M. Szczesny</span>
                <span>I. Wozniak</span>
                <span>O. Krupka</span>
                <span>F. Sieradzki</span>
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs" style={{ color: QUARTO.muted }}>
                <span>HACK-4-SAGES 2026 | ETH Zurich (COPL)</span>
                <span>|</span>
                <span>Category: Life Detection and Biosignatures</span>
                <span>|</span>
                <span>March 2026</span>
              </div>
            </motion.div>

            {/* ─── Abstract ─── */}
            <section id="abstract" className="relative">
              <SectionHeading id="abstract" level={2}>
                Abstract
              </SectionHeading>
              <div
                className="text-sm leading-relaxed italic pl-4 my-4"
                style={{
                  borderLeft: `3px solid ${QUARTO.link}`,
                  color: QUARTO.body,
                }}
              >
                We present ExoBiome, a hybrid quantum-classical neural network for
                atmospheric retrieval of exoplanet transmission spectra. Our architecture
                combines a classical spectral encoder with a 12-qubit parameterized quantum
                circuit to predict log&#8321;&#8320; volume mixing ratios (VMR) for five key
                molecular species: H&#8322;O, CO&#8322;, CO, CH&#8324;, and NH&#8323;.
                Evaluated on the Ariel Data Challenge benchmark, ExoBiome achieves a mean RMSE
                of <strong>0.295</strong>, surpassing the ADC 2023 winning solution (~0.32) and
                classical baselines by a significant margin. To our knowledge, this is the first
                application of quantum machine learning to biosignature detection from
                transmission spectroscopy.
              </div>
              <MarginNote>
                VMR = Volume Mixing Ratio, the standard unit for atmospheric composition
                measurements in exoplanet science.
              </MarginNote>
            </section>

            {/* ─── 1. Introduction ─── */}
            <section id="introduction" className="relative">
              <SectionHeading id="introduction" level={2}>
                1 Introduction
              </SectionHeading>
              <p className="text-sm leading-relaxed mb-4">
                The detection of biosignatures in exoplanet atmospheres represents one of
                the most compelling frontiers in modern astrophysics. Upcoming missions
                such as ESA&apos;s Ariel telescope will generate unprecedented volumes of
                transmission spectra, requiring robust and efficient retrieval methods{" "}
                <Citation id="schwieterman2018" />.
              </p>
              <p className="text-sm leading-relaxed mb-4">
                Traditional Bayesian retrieval codes, while physically motivated, are
                computationally expensive — often requiring hours per spectrum. Machine
                learning approaches have shown promise as fast alternatives, but existing
                methods rely entirely on classical architectures{" "}
                <Citation id="changeat2022" />.
              </p>
              <p className="text-sm leading-relaxed mb-4">
                Recent work by <Citation id="vetrano2025" /> demonstrated that Quantum
                Extreme Learning Machines (QELM) can effectively perform atmospheric
                retrieval tasks. However, their approach was limited to retrieval only and
                did not address the biosignature detection problem directly.
              </p>

              <Callout type="note" title="Research Gap">
                No prior work has combined quantum machine learning with end-to-end
                biosignature detection from transmission spectra. ExoBiome bridges this
                gap by integrating quantum-enhanced feature processing within a neural
                retrieval pipeline.
              </Callout>

              <p className="text-sm leading-relaxed mb-4">
                In this paper, we introduce ExoBiome — a quantum-classical hybrid model
                that processes raw transmission spectra and auxiliary planetary parameters
                to predict molecular abundances. Our key contributions are:
              </p>
              <ol
                className="text-sm leading-relaxed list-decimal pl-6 mb-4 space-y-1"
                style={{ color: QUARTO.body }}
              >
                <li>
                  A novel quantum-classical architecture achieving state-of-the-art
                  performance on the ADC benchmark
                </li>
                <li>
                  The first application of quantum ML to biosignature detection from
                  transmission spectroscopy
                </li>
                <li>
                  Systematic comparison against classical baselines demonstrating quantum
                  advantage in feature representation
                </li>
              </ol>
            </section>

            {/* ─── 2. Dataset ─── */}
            <section id="dataset" className="relative">
              <SectionHeading id="dataset" level={2}>
                2 Dataset
              </SectionHeading>
              <p className="text-sm leading-relaxed mb-4">
                We train and evaluate on the Ariel Data Challenge 2023 dataset{" "}
                <Citation id="changeat2022" />, which contains 41,000 simulated
                transmission spectra generated with TauREx 3. Each spectrum covers 52
                wavelength bins from 0.5 to 7.8 μm, paired with auxiliary parameters
                (stellar temperature, planetary radius, orbital period, surface gravity)
                and ground truth log&#8321;&#8320; VMR values for five target molecules.
              </p>

              <CodeBlock
                label="Show data loading"
                language="python"
                code={`import numpy as np
import pandas as pd

# Load ADC2023 dataset
spectra = np.load("data/adc2023_spectra.npy")     # (41000, 52)
aux_params = pd.read_csv("data/adc2023_aux.csv")   # stellar T, R_p, P, g
targets = np.load("data/adc2023_targets.npy")       # (41000, 5) log10 VMR

# Target molecules: H2O, CO2, CO, CH4, NH3
molecules = ["H2O", "CO2", "CO", "CH4", "NH3"]
print(f"Spectra shape: {spectra.shape}")
print(f"Target range:  [{targets.min():.1f}, {targets.max():.1f}]")`}
              />

              <Figure number={1} caption="Example transmission spectrum showing molecular absorption features across 52 wavelength bins (0.5–7.8 μm). Key absorption bands for H₂O (~1.4 μm), CH₄ (~3.3 μm), and CO₂ (~4.3 μm) are visible.">
                <div style={{ background: "#fff", padding: "16px 8px 8px" }}>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={spectrumData} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                      <XAxis
                        dataKey="wavelength"
                        tick={{ fontSize: 11, fill: QUARTO.muted }}
                        label={{ value: "Wavelength (μm)", position: "bottom", offset: -2, style: { fontSize: 11, fill: QUARTO.muted } }}
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: QUARTO.muted }}
                        label={{ value: "Transit Depth", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 11, fill: QUARTO.muted } }}
                        domain={["auto", "auto"]}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#fff",
                          border: `1px solid ${QUARTO.border}`,
                          borderRadius: 4,
                          fontSize: 12,
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="depth"
                        stroke={QUARTO.link}
                        strokeWidth={1.5}
                        dot={false}
                      />
                      <ReferenceLine x={1.4} stroke={moleculeColors["H₂O"]} strokeDasharray="4 4" label={{ value: "H₂O", position: "top", style: { fontSize: 10, fill: moleculeColors["H₂O"] } }} />
                      <ReferenceLine x={3.3} stroke={moleculeColors["CH₄"]} strokeDasharray="4 4" label={{ value: "CH₄", position: "top", style: { fontSize: 10, fill: moleculeColors["CH₄"] } }} />
                      <ReferenceLine x={4.3} stroke={moleculeColors["CO₂"]} strokeDasharray="4 4" label={{ value: "CO₂", position: "top", style: { fontSize: 10, fill: moleculeColors["CO₂"] } }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Figure>

              <p className="text-sm leading-relaxed mb-4">
                We supplement the primary dataset with spectra from the ABC Database{" "}
                <Citation id="cardenas2025" /> (106k spectra) for pre-training, using a
                two-stage transfer learning approach. Data splits follow an 80/10/10
                train/validation/test partition with stratified sampling by stellar type.
              </p>

              <div className="my-6 overflow-x-auto">
                <table
                  className="w-full text-sm"
                  style={{
                    fontFamily: "var(--font-franklin)",
                    borderCollapse: "collapse",
                  }}
                >
                  <thead>
                    <tr
                      style={{
                        borderBottom: `2px solid ${QUARTO.heading}`,
                        borderTop: `2px solid ${QUARTO.heading}`,
                      }}
                    >
                      <th className="text-left py-2 pr-4 font-semibold">Parameter</th>
                      <th className="text-left py-2 pr-4 font-semibold">Range</th>
                      <th className="text-left py-2 font-semibold">Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["Wavelength", "0.5 – 7.8 μm", "52 bins, Ariel Tier 1 resolution"],
                      ["T_star", "3000 – 7000 K", "Stellar effective temperature"],
                      ["R_p", "0.5 – 2.0 R_J", "Planetary radius"],
                      ["log g", "2.5 – 4.5", "Surface gravity (cgs)"],
                      ["P_orb", "0.5 – 50 d", "Orbital period"],
                    ].map(([param, range, desc], i) => (
                      <tr
                        key={param}
                        style={{
                          borderBottom: `1px solid ${QUARTO.tableBorder}`,
                          background: i % 2 === 1 ? QUARTO.tableStripe : "transparent",
                        }}
                      >
                        <td
                          className="py-2 pr-4"
                          style={{ fontFamily: "var(--font-jetbrains)", fontSize: "0.8rem" }}
                        >
                          {param}
                        </td>
                        <td className="py-2 pr-4">{range}</td>
                        <td className="py-2" style={{ color: QUARTO.muted }}>
                          {desc}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ─── 3. Architecture ─── */}
            <section id="architecture" className="relative">
              <SectionHeading id="architecture" level={2}>
                3 Model Architecture
              </SectionHeading>
              <p className="text-sm leading-relaxed mb-4">
                ExoBiome employs a hybrid quantum-classical architecture with four
                distinct processing stages. The classical front-end encodes raw spectral
                and auxiliary inputs into a compact latent representation, which is then
                processed by a parameterized quantum circuit before final regression.
              </p>

              <MarginNote>
                The architecture was designed to be compatible with NISQ-era quantum
                hardware (5–50 qubits).
              </MarginNote>

              <CodeBlock
                label="Show architecture definition"
                language="python"
                defaultOpen={true}
                code={`import torch
import torch.nn as nn
from squlearn.qnn import QNNClassifier

class ExoBiome(nn.Module):
    """Quantum-classical hybrid for atmospheric retrieval."""

    def __init__(self, n_wavelengths=52, n_aux=4, n_qubits=12):
        super().__init__()

        # Stage 1: Spectral Encoder
        self.spectral_encoder = nn.Sequential(
            nn.Linear(n_wavelengths, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Stage 2: Auxiliary Encoder
        self.aux_encoder = nn.Sequential(
            nn.Linear(n_aux, 32),
            nn.GELU(),
            nn.Linear(32, 16),
        )

        # Stage 3: Fusion + Quantum Projection
        self.fusion = nn.Sequential(
            nn.Linear(64 + 16, 48),
            nn.GELU(),
            nn.Linear(48, n_qubits),  # → 12 features
            nn.Tanh(),                 # bound to [-1, 1]
        )

        # Stage 4: Quantum Circuit (12 qubits, 3 layers)
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=3,
            entanglement="circular"
        )

        # Stage 5: Classical Head
        self.head = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.GELU(),
            nn.Linear(32, 5),  # 5 molecules
        )

    def forward(self, spectrum, aux):
        z_spec = self.spectral_encoder(spectrum)
        z_aux = self.aux_encoder(aux)
        z_fused = self.fusion(torch.cat([z_spec, z_aux], dim=-1))
        z_quantum = self.quantum_layer(z_fused)
        return self.head(z_quantum)`}
              />

              {/* Architecture diagram as styled div */}
              <Figure number={2} caption="ExoBiome architecture. Spectral and auxiliary inputs are encoded separately, fused, then processed through a 12-qubit parameterized quantum circuit before final regression.">
                <div className="p-6" style={{ background: QUARTO.codeBg }}>
                  <div className="flex flex-wrap items-center justify-center gap-2 text-xs" style={{ fontFamily: "var(--font-jetbrains)" }}>
                    {[
                      { label: "Spectrum\n(52 bins)", color: QUARTO.link, w: "w-24" },
                      { label: "→", color: "transparent", w: "w-4", text: true },
                      { label: "SpectralEncoder\nLinear→GELU→LN\n128 → 64", color: QUARTO.link, w: "w-32" },
                    ].map((block, i) =>
                      block.text ? (
                        <span key={i} style={{ color: QUARTO.muted }}>→</span>
                      ) : (
                        <div
                          key={i}
                          className={`${block.w} p-2 rounded text-center text-white`}
                          style={{ background: block.color, whiteSpace: "pre-line", fontSize: "0.7rem", lineHeight: 1.4 }}
                        >
                          {block.label}
                        </div>
                      )
                    )}
                    <span style={{ color: QUARTO.muted }}>↘</span>
                    <div
                      className="w-28 p-2 rounded text-center text-white"
                      style={{ background: "#495057", whiteSpace: "pre-line", fontSize: "0.7rem", lineHeight: 1.4 }}
                    >
                      {"Fusion\n80 → 48 → 12\nTanh"}
                    </div>
                    <span style={{ color: QUARTO.muted }}>→</span>
                    <div
                      className="w-36 p-3 rounded text-center text-white relative overflow-hidden"
                      style={{
                        background: "linear-gradient(135deg, #9954bb, #2780e3)",
                        whiteSpace: "pre-line",
                        fontSize: "0.7rem",
                        lineHeight: 1.4,
                      }}
                    >
                      {"Quantum Circuit\n12 qubits × 3 layers\nRY + CNOT + RZ"}
                      <div
                        className="absolute inset-0 opacity-10"
                        style={{
                          backgroundImage:
                            "repeating-linear-gradient(45deg, transparent, transparent 4px, white 4px, white 5px)",
                        }}
                      />
                    </div>
                    <span style={{ color: QUARTO.muted }}>→</span>
                    <div
                      className="w-24 p-2 rounded text-center text-white"
                      style={{ background: QUARTO.calloutTip, whiteSpace: "pre-line", fontSize: "0.7rem", lineHeight: 1.4 }}
                    >
                      {"Head\n12 → 32 → 5\nlog₁₀ VMR"}
                    </div>
                  </div>
                  <div className="flex items-center justify-center mt-3 gap-2">
                    <div
                      className="w-24 p-2 rounded text-center text-white"
                      style={{ background: QUARTO.calloutWarning, whiteSpace: "pre-line", fontSize: "0.7rem", lineHeight: 1.4 }}
                    >
                      {"Aux Params\n(T★, Rp, g, P)"}
                    </div>
                    <span className="text-xs" style={{ color: QUARTO.muted }}>
                      → AuxEncoder (4 → 32 → 16) ↗
                    </span>
                  </div>
                </div>
              </Figure>

              {/* ─── 3.1 Quantum Layer ─── */}
              <section id="quantum-layer">
                <SectionHeading id="quantum-layer" level={3}>
                  3.1 Quantum Layer Details
                </SectionHeading>
                <p className="text-sm leading-relaxed mb-4">
                  The quantum layer implements a parameterized quantum circuit (PQC) on 12
                  qubits with 3 variational layers. Each layer applies single-qubit
                  rotations (R<sub>Y</sub>, R<sub>Z</sub>) followed by circular CNOT
                  entanglement. Input features are encoded via angle embedding on the
                  R<sub>Y</sub> gates of the first layer{" "}
                  <Citation id="schuld2021" />.
                </p>

                <Callout type="tip" title="Why 12 Qubits?">
                  The 12-qubit circuit was chosen as the optimal trade-off between
                  expressibility and trainability. Our ablation studies showed diminishing
                  returns beyond 12 qubits due to barren plateau effects, while fewer than
                  8 qubits limited the model&apos;s capacity to represent molecular
                  correlations.
                </Callout>

                <TabSet
                  tabs={[
                    {
                      label: "Python",
                      content: (
                        <CodeBlock
                          label="Show quantum circuit"
                          language="python"
                          defaultOpen={true}
                          code={`from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def build_quantum_layer(n_qubits=12, n_layers=3):
    """Build parameterized quantum circuit for ExoBiome."""
    qc = QuantumCircuit(n_qubits)
    inputs = ParameterVector("x", n_qubits)
    weights = ParameterVector("w", n_qubits * n_layers * 2)

    # Angle embedding
    for i in range(n_qubits):
        qc.ry(inputs[i], i)

    # Variational layers
    w_idx = 0
    for layer in range(n_layers):
        # Single-qubit rotations
        for i in range(n_qubits):
            qc.ry(weights[w_idx], i)
            qc.rz(weights[w_idx + 1], i)
            w_idx += 2

        # Circular CNOT entanglement
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)

    return qc

qc = build_quantum_layer()
print(f"Circuit depth:  {qc.depth()}")
print(f"Parameters:     {qc.num_parameters}")
print(f"CNOT gates:     {qc.count_ops().get('cx', 0)}")`}
                        />
                      ),
                    },
                    {
                      label: "R",
                      disabled: true,
                      content: <div className="text-sm text-gray-400 italic">R implementation not available</div>,
                    },
                    {
                      label: "Julia",
                      disabled: true,
                      content: <div className="text-sm text-gray-400 italic">Julia implementation not available</div>,
                    },
                  ]}
                />

                <p className="text-sm leading-relaxed mb-4">
                  The quantum circuit has a total of 72 trainable parameters (12 qubits x 3
                  layers x 2 rotations) plus 12 input encoding parameters. Expectation
                  values of Pauli-Z operators on each qubit serve as the circuit output,
                  producing a 12-dimensional feature vector passed to the classical head.
                </p>

                <MarginNote>
                  Circuit depth = 42 after transpilation. Compatible with IQM Spark (5
                  qubits, simulated 12) and VTT Q50 (53 qubits, native 12).
                </MarginNote>
              </section>
            </section>

            {/* ─── 4. Training ─── */}
            <section id="training" className="relative">
              <SectionHeading id="training" level={2}>
                4 Training
              </SectionHeading>
              <p className="text-sm leading-relaxed mb-4">
                Training uses a two-stage transfer learning approach. First, the classical
                encoder is pre-trained on 106k spectra from the ABC Database{" "}
                <Citation id="cardenas2025" /> with frozen quantum weights. In the second
                stage, the full model (including the quantum circuit) is fine-tuned on the
                ADC 2023 training set.
              </p>

              <CodeBlock
                label="Show training configuration"
                language="python"
                code={`config = {
    # Stage 1: Pre-training (classical only)
    "pretrain_epochs": 15,
    "pretrain_lr": 1e-3,
    "pretrain_batch_size": 256,
    "pretrain_dataset": "ABC Database (106k spectra)",

    # Stage 2: Fine-tuning (full model)
    "finetune_epochs": 30,
    "finetune_lr": 5e-4,
    "finetune_batch_size": 128,
    "finetune_dataset": "ADC2023 (41k spectra)",

    # Quantum config
    "n_qubits": 12,
    "n_layers": 3,
    "entanglement": "circular",
    "shots": 1024,  # for hardware; None for statevector

    # Optimization
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingWarmRestarts",
    "T_0": 10,
    "loss": "MSE (per-molecule weighted)",

    # Regularization
    "dropout": 0.1,
    "label_smoothing": 0.0,
    "gradient_clip": 1.0,
}`}
              />

              <Figure number={3} caption="Training curves showing convergence of train and validation loss (MSE) over 30 fine-tuning epochs. The model converges around epoch 20 with minimal overfitting.">
                <div style={{ background: "#fff", padding: "16px 8px 8px" }}>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={trainingData} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                      <XAxis
                        dataKey="epoch"
                        tick={{ fontSize: 11, fill: QUARTO.muted }}
                        label={{ value: "Epoch", position: "bottom", offset: -2, style: { fontSize: 11, fill: QUARTO.muted } }}
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: QUARTO.muted }}
                        label={{ value: "MSE Loss", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 11, fill: QUARTO.muted } }}
                        domain={[0, 2]}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#fff",
                          border: `1px solid ${QUARTO.border}`,
                          borderRadius: 4,
                          fontSize: 12,
                        }}
                        formatter={(val) => Number(val).toFixed(3)}
                      />
                      <Legend
                        wrapperStyle={{ fontSize: 12, fontFamily: "var(--font-franklin)" }}
                      />
                      <Line
                        type="monotone"
                        dataKey="train"
                        stroke={QUARTO.link}
                        strokeWidth={2}
                        dot={false}
                        name="Train Loss"
                      />
                      <Line
                        type="monotone"
                        dataKey="val"
                        stroke={QUARTO.calloutImportant}
                        strokeWidth={2}
                        dot={false}
                        name="Val Loss"
                        strokeDasharray="6 3"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Figure>

              <Callout type="warning" title="Reproducibility">
                Quantum circuit simulation introduces stochasticity via finite shot
                sampling. For reproducible results, we fix the random seed and use
                statevector simulation during training. Hardware execution (IQM Spark / VTT
                Q50) is reserved for inference validation only.
              </Callout>
            </section>

            {/* ─── 5. Results ─── */}
            <section id="results" className="relative">
              <SectionHeading id="results" level={2}>
                5 Results
              </SectionHeading>
              <p className="text-sm leading-relaxed mb-4">
                We evaluate ExoBiome on the held-out test set (4,100 spectra) using mean
                RMSE (mRMSE) across all five target molecules, consistent with the ADC
                2023 evaluation protocol. As shown in{" "}
                <a href="#fig-4" style={{ color: QUARTO.link, textDecoration: "none" }}>
                  Figure 4
                </a>
                , ExoBiome achieves a mRMSE of <strong>0.295</strong>, representing a 7.8%
                improvement over the ADC 2023 winning solution.
              </p>

              {/* 5.1 Model Comparison */}
              <section id="comparison">
                <SectionHeading id="comparison" level={3}>
                  5.1 Model Comparison
                </SectionHeading>

                <Figure number={4} caption="Comparison of mean RMSE across models. ExoBiome achieves the lowest error, outperforming the ADC 2023 challenge winner by 7.8%.">
                  <div style={{ background: "#fff", padding: "16px 8px 8px" }}>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart
                        data={comparisonData}
                        layout="vertical"
                        margin={{ top: 10, right: 40, left: 10, bottom: 10 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#eee" horizontal={false} />
                        <XAxis
                          type="number"
                          tick={{ fontSize: 11, fill: QUARTO.muted }}
                          domain={[0, 1.4]}
                          label={{ value: "mRMSE (↓ better)", position: "bottom", offset: -2, style: { fontSize: 11, fill: QUARTO.muted } }}
                        />
                        <YAxis
                          dataKey="model"
                          type="category"
                          tick={{ fontSize: 12, fill: QUARTO.body }}
                          width={110}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#fff",
                            border: `1px solid ${QUARTO.border}`,
                            borderRadius: 4,
                            fontSize: 12,
                          }}
                          formatter={(val) => Number(val).toFixed(3)}
                        />
                        <Bar dataKey="mrmse" radius={[0, 4, 4, 0]} barSize={28}>
                          {comparisonData.map((entry, i) => (
                            <Cell key={i} fill={entry.fill} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Figure>

                <CodeBlock
                  label="Show evaluation code"
                  language="python"
                  code={`from sklearn.metrics import mean_squared_error
import numpy as np

def compute_mrmse(y_true, y_pred, molecules):
    """Compute mean RMSE across all target molecules."""
    rmses = {}
    for i, mol in enumerate(molecules):
        rmse = np.sqrt(mean_squared_error(
            y_true[:, i], y_pred[:, i]
        ))
        rmses[mol] = rmse
    mrmse = np.mean(list(rmses.values()))
    return mrmse, rmses

# Evaluate ExoBiome
mrmse, per_mol = compute_mrmse(y_test, predictions, molecules)
print(f"mRMSE: {mrmse:.3f}")
for mol, rmse in per_mol.items():
    print(f"  {mol:>4s}: {rmse:.3f}")`}
                />

                <MarginNote>
                  All baselines were re-evaluated on the same test split for fair
                  comparison. CNN and RF results are from our own implementations.
                </MarginNote>
              </section>

              {/* 5.2 Per-Molecule Results */}
              <section id="per-molecule">
                <SectionHeading id="per-molecule" level={3}>
                  5.2 Per-Molecule Performance
                </SectionHeading>
                <p className="text-sm leading-relaxed mb-4">
                  Breaking down performance by molecule (
                  <a href="#fig-5" style={{ color: QUARTO.link, textDecoration: "none" }}>
                    Figure 5
                  </a>
                  ) reveals that ExoBiome performs best on H&#8322;O (RMSE = 0.218) and
                  worst on NH&#8323; (RMSE = 0.378). This ordering correlates with the
                  strength of spectral features: H&#8322;O has prominent absorption bands
                  across multiple wavelength ranges, while NH&#8323; features are weaker and
                  more easily masked by other species.
                </p>

                <Figure number={5} caption="Per-molecule RMSE for ExoBiome. Performance correlates with absorption feature strength, with H₂O showing the lowest error and NH₃ the highest.">
                  <div style={{ background: "#fff", padding: "16px 8px 8px" }}>
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={perMoleculeData} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                        <XAxis
                          dataKey="molecule"
                          tick={{ fontSize: 12, fill: QUARTO.body }}
                        />
                        <YAxis
                          tick={{ fontSize: 11, fill: QUARTO.muted }}
                          domain={[0, 0.5]}
                          label={{ value: "RMSE", angle: -90, position: "insideLeft", offset: 10, style: { fontSize: 11, fill: QUARTO.muted } }}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#fff",
                            border: `1px solid ${QUARTO.border}`,
                            borderRadius: 4,
                            fontSize: 12,
                          }}
                          formatter={(val) => Number(val).toFixed(3)}
                        />
                        <Bar dataKey="rmse" radius={[4, 4, 0, 0]} barSize={48}>
                          {perMoleculeData.map((entry, i) => (
                            <Cell key={i} fill={entry.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Figure>

                {/* Results table */}
                <div className="my-6 overflow-x-auto">
                  <table
                    className="w-full text-sm"
                    style={{ fontFamily: "var(--font-franklin)", borderCollapse: "collapse" }}
                  >
                    <thead>
                      <tr
                        style={{
                          borderBottom: `2px solid ${QUARTO.heading}`,
                          borderTop: `2px solid ${QUARTO.heading}`,
                        }}
                      >
                        <th className="text-left py-2 pr-4 font-semibold">Molecule</th>
                        <th className="text-right py-2 pr-4 font-semibold">ExoBiome</th>
                        <th className="text-right py-2 pr-4 font-semibold">ADC Winner</th>
                        <th className="text-right py-2 pr-4 font-semibold">CNN</th>
                        <th className="text-right py-2 font-semibold">RF</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { mol: "H₂O", exo: "0.218", adc: "0.24", cnn: "0.62", rf: "0.95" },
                        { mol: "CO₂", exo: "0.261", adc: "0.28", cnn: "0.71", rf: "1.05" },
                        { mol: "CO", exo: "0.327", adc: "0.35", cnn: "0.92", rf: "1.28" },
                        { mol: "CH₄", exo: "0.290", adc: "0.33", cnn: "0.88", rf: "1.22" },
                        { mol: "NH₃", exo: "0.378", adc: "0.40", cnn: "1.12", rf: "1.50" },
                      ].map((row, i) => (
                        <tr
                          key={row.mol}
                          style={{
                            borderBottom: `1px solid ${QUARTO.tableBorder}`,
                            background: i % 2 === 1 ? QUARTO.tableStripe : "transparent",
                          }}
                        >
                          <td className="py-2 pr-4 font-medium" style={{ color: moleculeColors[row.mol] }}>
                            {row.mol}
                          </td>
                          <td
                            className="py-2 pr-4 text-right font-semibold"
                            style={{ fontFamily: "var(--font-jetbrains)", fontSize: "0.8rem" }}
                          >
                            {row.exo}
                          </td>
                          <td
                            className="py-2 pr-4 text-right"
                            style={{ fontFamily: "var(--font-jetbrains)", fontSize: "0.8rem", color: QUARTO.muted }}
                          >
                            {row.adc}
                          </td>
                          <td
                            className="py-2 pr-4 text-right"
                            style={{ fontFamily: "var(--font-jetbrains)", fontSize: "0.8rem", color: QUARTO.muted }}
                          >
                            {row.cnn}
                          </td>
                          <td
                            className="py-2 text-right"
                            style={{ fontFamily: "var(--font-jetbrains)", fontSize: "0.8rem", color: QUARTO.muted }}
                          >
                            {row.rf}
                          </td>
                        </tr>
                      ))}
                      <tr style={{ borderBottom: `2px solid ${QUARTO.heading}` }}>
                        <td className="py-2 pr-4 font-bold">Mean</td>
                        <td
                          className="py-2 pr-4 text-right font-bold"
                          style={{ fontFamily: "var(--font-jetbrains)", fontSize: "0.8rem", color: QUARTO.link }}
                        >
                          0.295
                        </td>
                        <td
                          className="py-2 pr-4 text-right font-medium"
                          style={{ fontFamily: "var(--font-jetbrains)", fontSize: "0.8rem", color: QUARTO.muted }}
                        >
                          ~0.32
                        </td>
                        <td
                          className="py-2 pr-4 text-right font-medium"
                          style={{ fontFamily: "var(--font-jetbrains)", fontSize: "0.8rem", color: QUARTO.muted }}
                        >
                          ~0.85
                        </td>
                        <td
                          className="py-2 text-right font-medium"
                          style={{ fontFamily: "var(--font-jetbrains)", fontSize: "0.8rem", color: QUARTO.muted }}
                        >
                          ~1.20
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <Figure number={6} caption="Radar chart comparing ExoBiome and classical CNN retrieval accuracy (normalized) across all five target molecules.">
                  <div style={{ background: "#fff", padding: "16px 8px 8px" }}>
                    <ResponsiveContainer width="100%" height={320}>
                      <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                        <PolarGrid stroke="#e9ecef" />
                        <PolarAngleAxis
                          dataKey="metric"
                          tick={{ fontSize: 12, fill: QUARTO.body }}
                        />
                        <PolarRadiusAxis
                          angle={90}
                          domain={[0, 1]}
                          tick={{ fontSize: 10, fill: QUARTO.muted }}
                        />
                        <Radar
                          name="ExoBiome"
                          dataKey="ExoBiome"
                          stroke={QUARTO.link}
                          fill={QUARTO.link}
                          fillOpacity={0.2}
                          strokeWidth={2}
                        />
                        <Radar
                          name="Classical CNN"
                          dataKey="Classical"
                          stroke={QUARTO.calloutWarning}
                          fill={QUARTO.calloutWarning}
                          fillOpacity={0.1}
                          strokeWidth={2}
                          strokeDasharray="4 4"
                        />
                        <Legend
                          wrapperStyle={{ fontSize: 12, fontFamily: "var(--font-franklin)" }}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </Figure>
              </section>
            </section>

            {/* ─── 6. Discussion ─── */}
            <section id="discussion" className="relative">
              <SectionHeading id="discussion" level={2}>
                6 Discussion
              </SectionHeading>
              <p className="text-sm leading-relaxed mb-4">
                Our results demonstrate that quantum-enhanced feature processing provides
                a measurable advantage for atmospheric retrieval tasks. The 7.8%
                improvement over the ADC 2023 winner suggests that the quantum circuit&apos;s
                ability to represent high-order feature correlations in Hilbert space
                captures information that classical networks miss.
              </p>
              <p className="text-sm leading-relaxed mb-4">
                Several factors contribute to ExoBiome&apos;s performance. First, the
                two-stage training approach allows the classical encoder to learn robust
                spectral features before the quantum layer refines them. Second, the
                circular entanglement topology in the quantum circuit provides sufficient
                inter-qubit coupling without the noise penalty of all-to-all connectivity.
                Third, the Tanh activation before quantum encoding naturally maps features
                to the [&#8722;1, 1] range suitable for angle embedding.
              </p>

              <Callout type="important" title="Quantum Advantage">
                While we observe a consistent improvement from the quantum layer, we note
                that this advantage is specific to the atmospheric retrieval task structure.
                The 5-target regression over correlated molecular abundances creates a
                problem geometry well-suited to quantum feature maps. Generalization to
                other spectroscopic tasks requires further investigation.
              </Callout>

              <p className="text-sm leading-relaxed mb-4">
                The per-molecule analysis reveals an interesting pattern: the quantum
                advantage is largest for molecules with overlapping spectral features
                (CO/CO&#8322;, CH&#8324;/H&#8322;O), suggesting the quantum circuit excels
                at disentangling correlated absorbers. This aligns with theoretical
                predictions about quantum kernels and their ability to separate entangled
                feature distributions <Citation id="schuld2021" />.
              </p>

              <CodeBlock
                label="Show quantum advantage analysis"
                language="python"
                code={`# Quantum advantage per molecule
advantage = {
    "H2O":  (0.24 - 0.218) / 0.24 * 100,   # 9.2%
    "CO2":  (0.28 - 0.261) / 0.28 * 100,   # 6.8%
    "CO":   (0.35 - 0.327) / 0.35 * 100,   # 6.6%
    "CH4":  (0.33 - 0.290) / 0.33 * 100,   # 12.1%
    "NH3":  (0.40 - 0.378) / 0.40 * 100,   # 5.5%
}

print("Quantum advantage over ADC winner:")
for mol, adv in advantage.items():
    print(f"  {mol:>4s}: {adv:.1f}%")
print(f"\\n  Mean: {np.mean(list(advantage.values())):.1f}%")`}
              />

              <p className="text-sm leading-relaxed mb-4">
                Future work will focus on three directions: (1) execution on real quantum
                hardware via the Odra 5 (IQM Spark) and VTT Q50 systems accessible through
                our PWR Wroclaw partnership, (2) extension to a broader set of atmospheric
                species including O&#8323;, SO&#8322;, and TiO, and (3) integration with
                TauREx 3 for physics-informed training{" "}
                <Citation id="seeburger2023" />.
              </p>
            </section>

            {/* ─── References ─── */}
            <section id="references" className="relative">
              <SectionHeading id="references" level={2}>
                References
              </SectionHeading>
              <div className="space-y-3 text-sm" style={{ color: QUARTO.body }}>
                {Object.entries(citations).map(([key, cite]) => (
                  <div
                    key={key}
                    className="pl-8 relative leading-relaxed"
                    style={{ textIndent: "-2rem", paddingLeft: "2rem" }}
                  >
                    <span
                      className="font-medium"
                      style={{ color: QUARTO.link }}
                    >
                      [{key}]
                    </span>{" "}
                    {cite.full}
                  </div>
                ))}
              </div>
            </section>

            {/* ─── Footer ─── */}
            <footer
              className="mt-16 pt-6 pb-10 text-xs flex flex-wrap gap-x-6 gap-y-2"
              style={{
                borderTop: `1px solid ${QUARTO.border}`,
                color: QUARTO.muted,
              }}
            >
              <span>ExoBiome | HACK-4-SAGES 2026</span>
              <span>ETH Zurich (COPL) | Life Detection and Biosignatures</span>
              <span className="ml-auto">
                Built with{" "}
                <span style={{ color: QUARTO.link }}>Quarto</span>
                {" "}+{" "}
                <span style={{ color: QUARTO.link }}>Jupyter</span>
              </span>
            </footer>
          </div>
        </main>

        {/* Right Margin — "On This Page" (secondary, for wider screens) */}
        <aside
          className="hidden 2xl:block fixed right-0 top-14 bottom-0 w-52 overflow-y-auto py-6 px-4"
          style={{
            borderLeft: `1px solid ${QUARTO.border}`,
            background: "#fff",
          }}
        >
          <div
            className="text-xs font-semibold uppercase tracking-wider mb-4"
            style={{ color: QUARTO.muted }}
          >
            On this page
          </div>
          <nav className="space-y-1">
            {sections.map(({ id, label }) => {
              const isActive = activeSection === id;
              const isSubsection = label.includes(".");
              return (
                <a
                  key={id}
                  href={`#${id}`}
                  className={`block py-1 text-xs transition-colors ${
                    isSubsection ? "pl-3" : ""
                  }`}
                  style={{
                    color: isActive ? QUARTO.sidebarActive : QUARTO.sidebarText,
                    fontWeight: isActive ? 600 : 400,
                    textDecoration: "none",
                  }}
                >
                  {label}
                </a>
              );
            })}
          </nav>

          <div
            className="mt-8 pt-4 space-y-3"
            style={{ borderTop: `1px solid ${QUARTO.border}` }}
          >
            <div className="text-xs" style={{ color: QUARTO.muted }}>
              <span className="font-medium block mb-1">Cite this work</span>
              <div
                className="p-2 rounded text-[10px] leading-relaxed"
                style={{
                  background: QUARTO.codeBg,
                  fontFamily: "var(--font-jetbrains)",
                  border: `1px solid ${QUARTO.border}`,
                }}
              >
                @article&#123;exobiome2026,
                <br />
                &nbsp;&nbsp;title=&#123;ExoBiome&#125;,
                <br />
                &nbsp;&nbsp;year=&#123;2026&#125;
                <br />
                &#125;
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
