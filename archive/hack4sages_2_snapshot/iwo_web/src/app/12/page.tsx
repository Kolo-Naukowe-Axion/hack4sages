"use client";

import { useState, useRef, useEffect } from "react";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import { motion, useInView, AnimatePresence } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  Tooltip,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";

const plexSans = IBM_Plex_Sans({
  weight: ["300", "400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-plex-sans",
});

const plexMono = IBM_Plex_Mono({
  weight: ["400", "500"],
  subsets: ["latin"],
  variable: "--font-plex-mono",
});

// --- DATA ---

const spectrumData = Array.from({ length: 52 }, (_, i) => {
  const wl = 0.8 + i * 0.09;
  const base = 0.0145 + Math.sin(wl * 1.2) * 0.001;
  const h2oFeature = wl > 1.3 && wl < 1.5 ? Math.exp(-((wl - 1.4) ** 2) / 0.005) * 0.0008 : 0;
  const co2Feature = wl > 2.6 && wl < 3.0 ? Math.exp(-((wl - 2.8) ** 2) / 0.008) * 0.0012 : 0;
  const ch4Feature = wl > 3.2 && wl < 3.5 ? Math.exp(-((wl - 3.3) ** 2) / 0.006) * 0.0006 : 0;
  const noise = (Math.random() - 0.5) * 0.0002;
  return {
    wavelength: +wl.toFixed(2),
    depth: +(base + h2oFeature + co2Feature + ch4Feature + noise).toFixed(5),
    error: +(0.00015 + Math.random() * 0.0001).toFixed(5),
  };
});

const modelComparisonData = [
  { model: "ExoBiome (Ours)", mrmse: 0.295, fill: "#3b82f6" },
  { model: "ADC Winner", mrmse: 0.32, fill: "#94a3b8" },
  { model: "CNN Baseline", mrmse: 0.85, fill: "#94a3b8" },
  { model: "Random Forest", mrmse: 1.2, fill: "#94a3b8" },
];

const perMoleculeData = [
  { molecule: "H₂O", rmse: 0.218, color: "#3b82f6" },
  { molecule: "CO₂", rmse: 0.261, color: "#8b5cf6" },
  { molecule: "CO", rmse: 0.327, color: "#f59e0b" },
  { molecule: "CH₄", rmse: 0.29, color: "#10b981" },
  { molecule: "NH₃", rmse: 0.378, color: "#ef4444" },
];

const quantumCircuitLayers = [
  { layer: 1, fidelity: 0.98 },
  { layer: 2, fidelity: 0.965 },
  { layer: 3, fidelity: 0.94 },
  { layer: 4, fidelity: 0.92 },
  { layer: 5, fidelity: 0.895 },
  { layer: 6, fidelity: 0.87 },
  { layer: 7, fidelity: 0.84 },
  { layer: 8, fidelity: 0.81 },
];

const trainingHistory = Array.from({ length: 7 }, (_, i) => ({
  epoch: i + 1,
  train: +(1.8 - 1.4 * (1 - Math.exp(-(i + 1) * 0.6))).toFixed(3),
  val: +(1.85 - 1.45 * (1 - Math.exp(-(i + 1) * 0.5))).toFixed(3),
}));

const residualData = Array.from({ length: 80 }, (_, i) => ({
  predicted: -4 + (i / 80) * 4,
  residual: (Math.random() - 0.5) * 0.8 + Math.sin(i * 0.15) * 0.1,
  size: 3 + Math.random() * 4,
}));

const tocSections = [
  { id: "header", label: "Title" },
  { id: "params", label: "Parameters" },
  { id: "spectrum", label: "Transmission Spectrum" },
  { id: "architecture", label: "Architecture" },
  { id: "quantum", label: "Quantum Circuit" },
  { id: "training", label: "Training" },
  { id: "benchmark", label: "Benchmark" },
  { id: "molecules", label: "Per-Molecule" },
  { id: "residuals", label: "Residuals" },
  { id: "conclusions", label: "Conclusions" },
];

// --- COMPONENTS ---

function CellWrapper({
  children,
  id,
  className = "",
}: {
  children: React.ReactNode;
  id?: string;
  className?: string;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-40px" });

  return (
    <motion.div
      ref={ref}
      id={id}
      initial={{ opacity: 0, y: 12 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 12 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={`mb-1 ${className}`}
    >
      {children}
    </motion.div>
  );
}

function CodeBlock({
  code,
  defaultOpen = false,
  language = "javascript",
}: {
  code: string;
  defaultOpen?: boolean;
  language?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="mt-3 mb-6">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-[12px] text-[#6e7781] hover:text-[#1a1a2e] transition-colors cursor-pointer"
        style={{ fontFamily: "var(--font-plex-mono)" }}
      >
        <svg
          width="10"
          height="10"
          viewBox="0 0 10 10"
          className={`transition-transform duration-200 ${open ? "rotate-90" : ""}`}
        >
          <path d="M3 1l5 4-5 4z" fill="currentColor" />
        </svg>
        <span>{language}</span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <pre
              className="mt-2 p-4 bg-[#f6f8fa] rounded-md text-[12.5px] leading-[1.6] overflow-x-auto border-l-[3px] border-[#3b82f6]"
              style={{ fontFamily: "var(--font-plex-mono)" }}
            >
              <code className="text-[#24292f]">{code}</code>
            </pre>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ObservableSlider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-3 py-1.5">
      <label
        className="text-[13px] text-[#57606a] whitespace-nowrap min-w-[140px]"
        style={{ fontFamily: "var(--font-plex-mono)" }}
      >
        {label}
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 h-[3px] appearance-none bg-[#d0d7de] rounded-full cursor-pointer accent-[#3b82f6]"
        style={{
          background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${((value - min) / (max - min)) * 100}%, #d0d7de ${((value - min) / (max - min)) * 100}%, #d0d7de 100%)`,
        }}
      />
      <span
        className="text-[13px] text-[#1a1a2e] font-medium tabular-nums min-w-[32px] text-right"
        style={{ fontFamily: "var(--font-plex-mono)" }}
      >
        {value}
      </span>
    </div>
  );
}

function ObservableSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex items-center gap-3 py-1.5">
      <label
        className="text-[13px] text-[#57606a] whitespace-nowrap min-w-[140px]"
        style={{ fontFamily: "var(--font-plex-mono)" }}
      >
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="text-[13px] px-2.5 py-1 border border-[#d0d7de] rounded-md bg-white text-[#1a1a2e] cursor-pointer hover:border-[#3b82f6] transition-colors outline-none focus:border-[#3b82f6] focus:ring-1 focus:ring-[#3b82f6]/20"
        style={{ fontFamily: "var(--font-plex-mono)" }}
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}

function Prose({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="text-[15px] leading-[1.72] text-[#24292f] max-w-none"
      style={{ fontFamily: "var(--font-plex-sans)" }}
    >
      {children}
    </div>
  );
}

function InlineCode({ children }: { children: React.ReactNode }) {
  return (
    <code
      className="text-[13px] px-[5px] py-[2px] bg-[#f0f3f6] rounded text-[#1a1a2e]"
      style={{ fontFamily: "var(--font-plex-mono)" }}
    >
      {children}
    </code>
  );
}

function SectionDivider() {
  return <hr className="border-0 border-t border-[#d8dee4] my-10" />;
}

function PinButton() {
  const [pinned, setPinned] = useState(false);
  return (
    <button
      onClick={() => setPinned(!pinned)}
      className={`opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-[#f0f3f6] ${pinned ? "!opacity-100" : ""}`}
      title={pinned ? "Unpin from sidebar" : "Pin to sidebar"}
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 16 16"
        fill={pinned ? "#3b82f6" : "none"}
        stroke={pinned ? "#3b82f6" : "#8b949e"}
        strokeWidth="1.5"
      >
        <path d="M4.5 2.5l7 0 0 3-2 2 0 3.5-1.5 2.5-1.5-2.5 0-3.5-2-2z" />
        <path d="M8 13.5l0 2" />
      </svg>
    </button>
  );
}

function ChartTooltipContent({
  label,
  value,
  unit,
}: {
  label: string;
  value: string;
  unit: string;
}) {
  return (
    <div
      className="bg-white border border-[#d0d7de] rounded-md px-3 py-2 shadow-sm"
      style={{ fontFamily: "var(--font-plex-mono)" }}
    >
      <p className="text-[11px] text-[#57606a]">{label}</p>
      <p className="text-[13px] text-[#1a1a2e] font-medium">
        {value} {unit}
      </p>
    </div>
  );
}

function ArchitectureBlock({
  label,
  sublabel,
  color,
  width = "w-full",
}: {
  label: string;
  sublabel: string;
  color: string;
  width?: string;
}) {
  return (
    <div
      className={`${width} px-4 py-3 rounded-md border text-center`}
      style={{ borderColor: color, backgroundColor: color + "08" }}
    >
      <div className="text-[13px] font-medium text-[#1a1a2e]">{label}</div>
      <div className="text-[11px] text-[#57606a] mt-0.5" style={{ fontFamily: "var(--font-plex-mono)" }}>
        {sublabel}
      </div>
    </div>
  );
}

function FlowArrow() {
  return (
    <div className="flex justify-center py-1.5">
      <svg width="16" height="20" viewBox="0 0 16 20">
        <path d="M8 0 L8 16 M3 12 L8 18 L13 12" stroke="#d0d7de" strokeWidth="1.5" fill="none" />
      </svg>
    </div>
  );
}

// --- MAIN PAGE ---

export default function Page() {
  const [numQubits, setNumQubits] = useState(12);
  const [dataset, setDataset] = useState("ADC 2023");
  const [architecture, setArchitecture] = useState("SpectralEncoder + AuxEncoder");
  const [activeSection, setActiveSection] = useState("header");

  const handleScroll = () => {
    for (const section of tocSections) {
      const el = document.getElementById(section.id);
      if (el) {
        const rect = el.getBoundingClientRect();
        if (rect.top < 200 && rect.bottom > 0) {
          setActiveSection(section.id);
        }
      }
    }
  };

  return (
    <div
      className={`${plexSans.variable} ${plexMono.variable} min-h-screen bg-white`}
      onScroll={handleScroll}
      style={{ fontFamily: "var(--font-plex-sans)" }}
    >
      {/* TOP BAR */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-sm border-b border-[#d8dee4]">
        <div className="flex items-center justify-between h-[48px] px-5">
          <div className="flex items-center gap-3">
            {/* Observable-style logo */}
            <div className="flex items-center gap-2">
              <svg width="22" height="22" viewBox="0 0 22 22">
                <circle cx="11" cy="11" r="9" stroke="#1a1a2e" strokeWidth="2" fill="none" />
                <circle cx="11" cy="11" r="3.5" fill="#3b82f6" />
              </svg>
              <span className="text-[14px] font-semibold text-[#1a1a2e] tracking-[-0.01em]">
                ExoBiome
              </span>
            </div>
            <span className="text-[#d0d7de]">/</span>
            <span className="text-[13px] text-[#57606a]">Quantum Biosignature Detection</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className="w-[26px] h-[26px] rounded-full bg-gradient-to-br from-[#3b82f6] to-[#8b5cf6] flex items-center justify-center">
                <span className="text-[11px] font-medium text-white">MS</span>
              </div>
              <span className="text-[12px] text-[#57606a]">Michał Szczesny</span>
            </div>
            <button className="text-[12px] px-3 py-1 bg-[#3b82f6] text-white rounded-md hover:bg-[#2563eb] transition-colors font-medium">
              Publish
            </button>
            <button className="text-[12px] px-3 py-1 border border-[#d0d7de] text-[#57606a] rounded-md hover:border-[#8b949e] transition-colors">
              Fork
            </button>
          </div>
        </div>
      </header>

      <div className="flex pt-[48px]">
        {/* LEFT SIDEBAR — TOC */}
        <nav className="fixed left-0 top-[48px] bottom-0 w-[200px] bg-[#f6f8fa] border-r border-[#d8dee4] py-6 px-4 overflow-y-auto hidden lg:block">
          <div className="text-[11px] font-semibold text-[#57606a] uppercase tracking-wider mb-3">
            Contents
          </div>
          <ul className="space-y-0.5">
            {tocSections.map((section) => (
              <li key={section.id}>
                <a
                  href={`#${section.id}`}
                  onClick={(e) => {
                    e.preventDefault();
                    document.getElementById(section.id)?.scrollIntoView({ behavior: "smooth" });
                  }}
                  className={`block text-[12.5px] py-1 px-2 rounded transition-colors ${
                    activeSection === section.id
                      ? "text-[#3b82f6] bg-[#3b82f6]/8 font-medium"
                      : "text-[#57606a] hover:text-[#1a1a2e] hover:bg-[#eaeef2]"
                  }`}
                >
                  {section.label}
                </a>
              </li>
            ))}
          </ul>

          <div className="mt-8 pt-4 border-t border-[#d8dee4]">
            <div className="text-[11px] font-semibold text-[#57606a] uppercase tracking-wider mb-2">
              Runtime
            </div>
            <div className="text-[11px] text-[#8b949e] space-y-1" style={{ fontFamily: "var(--font-plex-mono)" }}>
              <div>Cells: 14</div>
              <div>Status: ✓ idle</div>
              <div>v0.6.1</div>
            </div>
          </div>
        </nav>

        {/* MAIN CONTENT */}
        <main className="flex-1 lg:ml-[200px] max-w-[820px] mx-auto px-6 lg:px-10 py-10 pb-32">
          {/* TITLE */}
          <CellWrapper id="header">
            <div className="group relative">
              <div className="absolute right-0 top-0">
                <PinButton />
              </div>
              <div className="text-[12px] text-[#8b949e] mb-2" style={{ fontFamily: "var(--font-plex-mono)" }}>
                HACK-4-SAGES 2026 · ETH Zurich
              </div>
              <h1 className="text-[32px] font-bold text-[#1a1a2e] leading-[1.2] tracking-[-0.02em]">
                Quantum Biosignature Detection from Exoplanet Transmission Spectra
              </h1>
              <p className="mt-3 text-[15px] text-[#57606a] leading-[1.6]">
                Can a 12-qubit quantum circuit identify signs of life in atmospheric spectra? We built a
                hybrid quantum-classical neural network that predicts molecular abundances (log₁₀ VMR)
                for five key biosignature gases — achieving state-of-the-art accuracy on the Ariel Data
                Challenge 2023 benchmark.
              </p>
              <div className="flex items-center gap-4 mt-4 text-[12px] text-[#8b949e]" style={{ fontFamily: "var(--font-plex-mono)" }}>
                <span>March 12, 2026</span>
                <span>·</span>
                <span>Michał Szczesny et al.</span>
                <span>·</span>
                <span className="text-[#3b82f6]">Apache 2.0</span>
              </div>
            </div>
          </CellWrapper>

          <SectionDivider />

          {/* INTERACTIVE PARAMETERS */}
          <CellWrapper id="params">
            <div className="group relative">
              <div className="absolute right-0 top-0">
                <PinButton />
              </div>
              <div className="bg-[#f6f8fa] border border-[#d8dee4] rounded-lg p-5">
                <div className="text-[11px] font-semibold text-[#57606a] uppercase tracking-wider mb-3">
                  viewof parameters
                </div>
                <ObservableSlider
                  label="num_qubits"
                  value={numQubits}
                  min={4}
                  max={20}
                  onChange={setNumQubits}
                />
                <ObservableSelect
                  label="dataset"
                  value={dataset}
                  options={["ADC 2023", "ABC Database", "MultiREx", "JWST Real"]}
                  onChange={setDataset}
                />
                <ObservableSelect
                  label="architecture"
                  value={architecture}
                  options={[
                    "SpectralEncoder + AuxEncoder",
                    "SpectralEncoder Only",
                    "CNN Baseline",
                    "Random Forest",
                  ]}
                  onChange={setArchitecture}
                />
              </div>
            </div>

            <CodeBlock
              code={`viewof num_qubits = Inputs.range([4, 20], {
  value: ${numQubits}, step: 1, label: "Number of qubits"
})

viewof dataset = Inputs.select(
  ["ADC 2023", "ABC Database", "MultiREx", "JWST Real"],
  { value: "${dataset}", label: "Dataset" }
)

viewof architecture = Inputs.select(
  ["SpectralEncoder + AuxEncoder", "SpectralEncoder Only", "CNN Baseline", "Random Forest"],
  { value: "${architecture}", label: "Architecture" }
)`}
            />
          </CellWrapper>

          <SectionDivider />

          {/* TRANSMISSION SPECTRUM */}
          <CellWrapper id="spectrum">
            <div className="group relative">
              <div className="absolute right-0 top-0">
                <PinButton />
              </div>
              <Prose>
                <h2 className="text-[20px] font-semibold text-[#1a1a2e] mb-1 tracking-[-0.01em]">
                  Input: Transmission Spectrum
                </h2>
                <p className="text-[14px] text-[#57606a] mb-5">
                  Each observation captures how starlight filters through an exoplanet atmosphere.
                  Molecular absorption creates characteristic dips at specific wavelengths — the
                  fingerprints we teach the model to read.
                </p>
              </Prose>

              {/* Output above code — Observable style */}
              <div className="bg-white border border-[#d8dee4] rounded-lg p-5 mb-1">
                <div className="flex items-center justify-between mb-4">
                  <div className="text-[11px] text-[#8b949e] uppercase tracking-wider" style={{ fontFamily: "var(--font-plex-mono)" }}>
                    Transit Depth vs. Wavelength
                  </div>
                  <div className="flex items-center gap-4 text-[11px] text-[#8b949e]" style={{ fontFamily: "var(--font-plex-mono)" }}>
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-[2px] bg-[#3b82f6] inline-block rounded" />
                      observed
                    </span>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={spectrumData} margin={{ top: 5, right: 10, bottom: 25, left: 15 }}>
                    <defs>
                      <linearGradient id="spectrumGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.12} />
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.01} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="wavelength"
                      tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" }}
                      axisLine={{ stroke: "#d0d7de" }}
                      tickLine={false}
                      label={{
                        value: "Wavelength (μm)",
                        position: "bottom",
                        offset: 10,
                        style: { fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" },
                      }}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" }}
                      axisLine={{ stroke: "#d0d7de" }}
                      tickLine={false}
                      domain={["auto", "auto"]}
                      tickFormatter={(v: number) => v.toFixed(4)}
                      label={{
                        value: "Transit depth",
                        angle: -90,
                        position: "insideLeft",
                        offset: -5,
                        style: { fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" },
                      }}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <ChartTooltipContent
                              label={`λ = ${payload[0].payload.wavelength} μm`}
                              value={payload[0].payload.depth.toFixed(5)}
                              unit="transit depth"
                            />
                          );
                        }
                        return null;
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="depth"
                      stroke="#3b82f6"
                      strokeWidth={1.5}
                      fill="url(#spectrumGrad)"
                      dot={false}
                      activeDot={{ r: 3, stroke: "#3b82f6", strokeWidth: 1.5, fill: "white" }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <CodeBlock
              code={`spectrum = FileAttachment("transit_spectrum.csv").csv({ typed: true })

Plot.plot({
  marks: [
    Plot.areaY(spectrum, { x: "wavelength", y: "depth", fill: "#3b82f6", fillOpacity: 0.1 }),
    Plot.lineY(spectrum, { x: "wavelength", y: "depth", stroke: "#3b82f6", strokeWidth: 1.5 })
  ],
  x: { label: "Wavelength (μm)" },
  y: { label: "Transit depth", grid: false },
  width: 720, height: 280
})`}
            />
          </CellWrapper>

          <SectionDivider />

          {/* ARCHITECTURE */}
          <CellWrapper id="architecture">
            <div className="group relative">
              <div className="absolute right-0 top-0">
                <PinButton />
              </div>
              <Prose>
                <h2 className="text-[20px] font-semibold text-[#1a1a2e] mb-1 tracking-[-0.01em]">
                  Model Architecture
                </h2>
                <p className="text-[14px] text-[#57606a] mb-5">
                  A hybrid pipeline fuses spectral features with auxiliary planet metadata, processes them
                  through a parameterized quantum circuit, and outputs log₁₀ volume mixing ratios for each
                  target molecule.
                </p>
              </Prose>

              <div className="bg-white border border-[#d8dee4] rounded-lg p-6 mb-1">
                <div className="max-w-[480px] mx-auto">
                  <div className="grid grid-cols-2 gap-3">
                    <ArchitectureBlock
                      label="SpectralEncoder"
                      sublabel="Conv1D → 52 bins → 64d"
                      color="#3b82f6"
                    />
                    <ArchitectureBlock
                      label="AuxEncoder"
                      sublabel="MLP → 6 features → 32d"
                      color="#8b5cf6"
                    />
                  </div>
                  <FlowArrow />
                  <ArchitectureBlock
                    label="Fusion Layer"
                    sublabel="concat → Linear(96, 64) → ReLU → Dropout(0.15)"
                    color="#6366f1"
                  />
                  <FlowArrow />
                  <ArchitectureBlock
                    label={`Quantum Circuit — ${numQubits} qubits`}
                    sublabel={`RY encoding → ${numQubits} entangling layers → Pauli-Z measurement`}
                    color="#f59e0b"
                  />
                  <FlowArrow />
                  <ArchitectureBlock
                    label="Regression Head"
                    sublabel="Linear(12, 32) → ReLU → Linear(32, 5)"
                    color="#10b981"
                  />
                  <FlowArrow />
                  <div className="grid grid-cols-5 gap-1.5">
                    {["H₂O", "CO₂", "CO", "CH₄", "NH₃"].map((mol, i) => (
                      <div
                        key={mol}
                        className="text-center py-2 rounded border text-[12px] font-medium"
                        style={{
                          borderColor: perMoleculeData[i].color,
                          color: perMoleculeData[i].color,
                          backgroundColor: perMoleculeData[i].color + "08",
                          fontFamily: "var(--font-plex-mono)",
                        }}
                      >
                        {mol}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <CodeBlock
              code={`architecture = {
  spectral_encoder: "Conv1D → 52 bins → 64d",
  aux_encoder: "MLP → 6 features → 32d",
  fusion: "concat → Linear(96, 64) → ReLU → Dropout(0.15)",
  quantum_circuit: \`RY encoding → \${num_qubits} entangling layers → Pauli-Z\`,
  regression_head: "Linear(12, 32) → ReLU → Linear(32, 5)",
  targets: ["H₂O", "CO₂", "CO", "CH₄", "NH₃"]
}`}
            />
          </CellWrapper>

          <SectionDivider />

          {/* QUANTUM CIRCUIT */}
          <CellWrapper id="quantum">
            <div className="group relative">
              <div className="absolute right-0 top-0">
                <PinButton />
              </div>
              <Prose>
                <h2 className="text-[20px] font-semibold text-[#1a1a2e] mb-1 tracking-[-0.01em]">
                  Quantum Circuit Fidelity
                </h2>
                <p className="text-[14px] text-[#57606a] mb-5">
                  Gate fidelity decreases with circuit depth. We use {numQubits} qubits with 8 variational
                  layers — a configuration that balances expressiveness against noise accumulation on
                  current NISQ hardware.
                </p>
              </Prose>

              <div className="bg-white border border-[#d8dee4] rounded-lg p-5 mb-1">
                <div className="flex items-center justify-between mb-4">
                  <div className="text-[11px] text-[#8b949e] uppercase tracking-wider" style={{ fontFamily: "var(--font-plex-mono)" }}>
                    Estimated fidelity per layer
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={quantumCircuitLayers} margin={{ top: 5, right: 10, bottom: 25, left: 15 }}>
                    <XAxis
                      dataKey="layer"
                      tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" }}
                      axisLine={{ stroke: "#d0d7de" }}
                      tickLine={false}
                      label={{
                        value: "Circuit layer",
                        position: "bottom",
                        offset: 10,
                        style: { fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" },
                      }}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" }}
                      axisLine={{ stroke: "#d0d7de" }}
                      tickLine={false}
                      domain={[0.75, 1]}
                      tickFormatter={(v: number) => v.toFixed(2)}
                      label={{
                        value: "Fidelity",
                        angle: -90,
                        position: "insideLeft",
                        offset: -5,
                        style: { fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" },
                      }}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <ChartTooltipContent
                              label={`Layer ${payload[0].payload.layer}`}
                              value={payload[0].payload.fidelity.toFixed(3)}
                              unit="fidelity"
                            />
                          );
                        }
                        return null;
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="fidelity"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      dot={{ r: 3, fill: "white", stroke: "#f59e0b", strokeWidth: 2 }}
                      activeDot={{ r: 4, stroke: "#f59e0b", strokeWidth: 2, fill: "white" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Quantum circuit ASCII diagram */}
              <div className="mt-4 bg-[#f6f8fa] border border-[#d8dee4] rounded-lg p-5">
                <div className="text-[11px] text-[#8b949e] uppercase tracking-wider mb-3" style={{ fontFamily: "var(--font-plex-mono)" }}>
                  Circuit structure (simplified)
                </div>
                <pre
                  className="text-[12px] leading-[1.8] text-[#24292f] overflow-x-auto"
                  style={{ fontFamily: "var(--font-plex-mono)" }}
                >
{`q₀: ──RY(θ₀)──●──────────RY(θ₁₂)──●──────────── ⟨Z⟩
                │                     │
q₁: ──RY(θ₁)──X──●───────RY(θ₁₃)──X──●─────────── ⟨Z⟩
                   │                    │
q₂: ──RY(θ₂)─────X──●────RY(θ₁₄)─────X──●──────── ⟨Z⟩
                      │                    │
  ⋮                   ⋮                    ⋮
                      │                    │
q₁₁:─RY(θ₁₁)────────X───RY(θ₂₃)─────────X──────── ⟨Z⟩`}
                </pre>
              </div>
            </div>

            <CodeBlock
              code={`circuit = QuantumCircuit(num_qubits)
for i in range(num_qubits):
    circuit.ry(Parameter(f"θ_{i}"), i)
for layer in range(n_layers):
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
    for i in range(num_qubits):
        circuit.ry(Parameter(f"θ_{layer}_{i}"), i)
measurements = [ExpectationValue(PauliZ(i)) for i in range(num_qubits)]`}
              language="python"
            />
          </CellWrapper>

          <SectionDivider />

          {/* TRAINING */}
          <CellWrapper id="training">
            <div className="group relative">
              <div className="absolute right-0 top-0">
                <PinButton />
              </div>
              <Prose>
                <h2 className="text-[20px] font-semibold text-[#1a1a2e] mb-1 tracking-[-0.01em]">
                  Training Dynamics
                </h2>
                <p className="text-[14px] text-[#57606a] mb-5">
                  Trained for 7 epochs on the {dataset} dataset with Adam optimizer (lr=1e-3, weight
                  decay=1e-4). The model converges quickly — validation loss plateaus by epoch 5, with
                  the best checkpoint at epoch 6.
                </p>
              </Prose>

              <div className="bg-white border border-[#d8dee4] rounded-lg p-5 mb-1">
                <div className="flex items-center justify-between mb-4">
                  <div className="text-[11px] text-[#8b949e] uppercase tracking-wider" style={{ fontFamily: "var(--font-plex-mono)" }}>
                    Loss vs. Epoch
                  </div>
                  <div className="flex items-center gap-4 text-[11px] text-[#8b949e]" style={{ fontFamily: "var(--font-plex-mono)" }}>
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-[2px] bg-[#3b82f6] inline-block rounded" />
                      train
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-[2px] bg-[#ef4444] inline-block rounded" />
                      validation
                    </span>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={trainingHistory} margin={{ top: 5, right: 10, bottom: 25, left: 15 }}>
                    <XAxis
                      dataKey="epoch"
                      tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" }}
                      axisLine={{ stroke: "#d0d7de" }}
                      tickLine={false}
                      label={{
                        value: "Epoch",
                        position: "bottom",
                        offset: 10,
                        style: { fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" },
                      }}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" }}
                      axisLine={{ stroke: "#d0d7de" }}
                      tickLine={false}
                      domain={[0, 2]}
                      label={{
                        value: "Loss",
                        angle: -90,
                        position: "insideLeft",
                        offset: -5,
                        style: { fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" },
                      }}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <div className="bg-white border border-[#d0d7de] rounded-md px-3 py-2 shadow-sm" style={{ fontFamily: "var(--font-plex-mono)" }}>
                              <p className="text-[11px] text-[#57606a]">Epoch {payload[0].payload.epoch}</p>
                              <p className="text-[12px] text-[#3b82f6]">train: {payload[0].payload.train}</p>
                              <p className="text-[12px] text-[#ef4444]">val: {payload[0].payload.val}</p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="train"
                      stroke="#3b82f6"
                      strokeWidth={1.5}
                      dot={{ r: 3, fill: "white", stroke: "#3b82f6", strokeWidth: 1.5 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="val"
                      stroke="#ef4444"
                      strokeWidth={1.5}
                      dot={{ r: 3, fill: "white", stroke: "#ef4444", strokeWidth: 1.5 }}
                      strokeDasharray="4 3"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <CodeBlock
              code={`optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7)
loss_fn = nn.MSELoss()

for epoch in range(7):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
    val_loss = evaluate(model, val_loader, loss_fn)
    scheduler.step()
    print(f"Epoch {epoch+1}: train={train_loss:.3f}, val={val_loss:.3f}")`}
              language="python"
            />
          </CellWrapper>

          <SectionDivider />

          {/* BENCHMARK */}
          <CellWrapper id="benchmark">
            <div className="group relative">
              <div className="absolute right-0 top-0">
                <PinButton />
              </div>
              <Prose>
                <h2 className="text-[20px] font-semibold text-[#1a1a2e] mb-1 tracking-[-0.01em]">
                  Benchmark: Model Comparison
                </h2>
                <p className="text-[14px] text-[#57606a] mb-5">
                  ExoBiome achieves a mean RMSE of <InlineCode>0.295</InlineCode> across all five target
                  molecules — surpassing the ADC 2023 competition winner (<InlineCode>~0.32</InlineCode>)
                  and classical baselines by a wide margin.
                </p>
              </Prose>

              <div className="bg-white border border-[#d8dee4] rounded-lg p-5 mb-1">
                <div className="flex items-center justify-between mb-4">
                  <div className="text-[11px] text-[#8b949e] uppercase tracking-wider" style={{ fontFamily: "var(--font-plex-mono)" }}>
                    Mean RMSE (lower is better)
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart
                    data={modelComparisonData}
                    layout="vertical"
                    margin={{ top: 0, right: 40, bottom: 0, left: 10 }}
                  >
                    <XAxis
                      type="number"
                      tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" }}
                      axisLine={{ stroke: "#d0d7de" }}
                      tickLine={false}
                      domain={[0, 1.4]}
                    />
                    <YAxis
                      type="category"
                      dataKey="model"
                      tick={{ fontSize: 12, fill: "#24292f", fontFamily: "var(--font-plex-sans)" }}
                      axisLine={false}
                      tickLine={false}
                      width={140}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <ChartTooltipContent
                              label={payload[0].payload.model}
                              value={payload[0].payload.mrmse.toFixed(3)}
                              unit="mRMSE"
                            />
                          );
                        }
                        return null;
                      }}
                    />
                    <Bar dataKey="mrmse" radius={[0, 3, 3, 0]} barSize={22}>
                      {modelComparisonData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>

                {/* Inline result highlight */}
                <div className="mt-4 pt-4 border-t border-[#eaeef2] flex items-center gap-6">
                  <div>
                    <div className="text-[11px] text-[#8b949e] uppercase tracking-wider" style={{ fontFamily: "var(--font-plex-mono)" }}>
                      Our result
                    </div>
                    <div className="text-[28px] font-bold text-[#3b82f6] tracking-[-0.02em]">0.295</div>
                    <div className="text-[12px] text-[#57606a]">mRMSE</div>
                  </div>
                  <div className="h-12 w-px bg-[#eaeef2]" />
                  <div>
                    <div className="text-[11px] text-[#8b949e] uppercase tracking-wider" style={{ fontFamily: "var(--font-plex-mono)" }}>
                      vs. ADC Winner
                    </div>
                    <div className="text-[28px] font-bold text-[#10b981] tracking-[-0.02em]">-7.8%</div>
                    <div className="text-[12px] text-[#57606a]">improvement</div>
                  </div>
                  <div className="h-12 w-px bg-[#eaeef2]" />
                  <div>
                    <div className="text-[11px] text-[#8b949e] uppercase tracking-wider" style={{ fontFamily: "var(--font-plex-mono)" }}>
                      vs. CNN
                    </div>
                    <div className="text-[28px] font-bold text-[#10b981] tracking-[-0.02em]">-65%</div>
                    <div className="text-[12px] text-[#57606a]">improvement</div>
                  </div>
                </div>
              </div>
            </div>

            <CodeBlock
              code={`comparison = [
  { model: "ExoBiome (Ours)", mrmse: 0.295 },
  { model: "ADC Winner",      mrmse: 0.32  },
  { model: "CNN Baseline",    mrmse: 0.85  },
  { model: "Random Forest",   mrmse: 1.20  }
]

Plot.plot({
  marks: [
    Plot.barX(comparison, {
      y: "model", x: "mrmse",
      fill: d => d.model.includes("Ours") ? "#3b82f6" : "#94a3b8",
      sort: { y: "x" }
    }),
    Plot.text(comparison, {
      y: "model", x: "mrmse",
      text: d => d.mrmse.toFixed(3), dx: 24
    })
  ],
  x: { label: "mRMSE →", domain: [0, 1.4] },
  marginLeft: 140
})`}
            />
          </CellWrapper>

          <SectionDivider />

          {/* PER-MOLECULE */}
          <CellWrapper id="molecules">
            <div className="group relative">
              <div className="absolute right-0 top-0">
                <PinButton />
              </div>
              <Prose>
                <h2 className="text-[20px] font-semibold text-[#1a1a2e] mb-1 tracking-[-0.01em]">
                  Per-Molecule Performance
                </h2>
                <p className="text-[14px] text-[#57606a] mb-5">
                  H₂O is the easiest target (strong, broad absorption features). NH₃ is hardest — its
                  weaker spectral signature overlaps with CH₄ and demands precise quantum feature
                  extraction.
                </p>
              </Prose>

              <div className="bg-white border border-[#d8dee4] rounded-lg p-5 mb-1">
                <div className="flex items-center justify-between mb-4">
                  <div className="text-[11px] text-[#8b949e] uppercase tracking-wider" style={{ fontFamily: "var(--font-plex-mono)" }}>
                    RMSE by molecule
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={perMoleculeData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                    <XAxis
                      dataKey="molecule"
                      tick={{ fontSize: 12, fill: "#24292f", fontFamily: "var(--font-plex-mono)" }}
                      axisLine={{ stroke: "#d0d7de" }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" }}
                      axisLine={{ stroke: "#d0d7de" }}
                      tickLine={false}
                      domain={[0, 0.5]}
                      label={{
                        value: "RMSE",
                        angle: -90,
                        position: "insideLeft",
                        offset: -5,
                        style: { fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" },
                      }}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <ChartTooltipContent
                              label={payload[0].payload.molecule}
                              value={payload[0].payload.rmse.toFixed(3)}
                              unit="RMSE"
                            />
                          );
                        }
                        return null;
                      }}
                    />
                    <Bar dataKey="rmse" radius={[3, 3, 0, 0]} barSize={48}>
                      {perMoleculeData.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>

                {/* Molecule cards */}
                <div className="grid grid-cols-5 gap-2 mt-5 pt-4 border-t border-[#eaeef2]">
                  {perMoleculeData.map((mol) => (
                    <div key={mol.molecule} className="text-center">
                      <div
                        className="text-[18px] font-bold tracking-[-0.01em]"
                        style={{ color: mol.color }}
                      >
                        {mol.rmse}
                      </div>
                      <div className="text-[11px] text-[#8b949e] mt-0.5" style={{ fontFamily: "var(--font-plex-mono)" }}>
                        {mol.molecule}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <CodeBlock
              code={`molecules = ["H₂O", "CO₂", "CO", "CH₄", "NH₃"]
rmse_scores = [0.218, 0.261, 0.327, 0.290, 0.378]

per_molecule = pd.DataFrame({
  "molecule": molecules,
  "rmse": rmse_scores
})

Plot.plot({
  marks: [
    Plot.barY(per_molecule, {
      x: "molecule", y: "rmse",
      fill: "molecule",
      sort: { x: "y" }
    }),
    Plot.ruleY([0])
  ],
  color: { legend: false },
  y: { label: "RMSE", domain: [0, 0.5] }
})`}
              language="python"
            />
          </CellWrapper>

          <SectionDivider />

          {/* RESIDUALS */}
          <CellWrapper id="residuals">
            <div className="group relative">
              <div className="absolute right-0 top-0">
                <PinButton />
              </div>
              <Prose>
                <h2 className="text-[20px] font-semibold text-[#1a1a2e] mb-1 tracking-[-0.01em]">
                  Residual Analysis
                </h2>
                <p className="text-[14px] text-[#57606a] mb-5">
                  Residuals are centered around zero with no systematic bias across the prediction range.
                  The model performs uniformly well across low-abundance and high-abundance molecular
                  targets.
                </p>
              </Prose>

              <div className="bg-white border border-[#d8dee4] rounded-lg p-5 mb-1">
                <div className="flex items-center justify-between mb-4">
                  <div className="text-[11px] text-[#8b949e] uppercase tracking-wider" style={{ fontFamily: "var(--font-plex-mono)" }}>
                    Predicted vs. Residual (holdout set)
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={240}>
                  <ScatterChart margin={{ top: 5, right: 10, bottom: 25, left: 15 }}>
                    <XAxis
                      type="number"
                      dataKey="predicted"
                      tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" }}
                      axisLine={{ stroke: "#d0d7de" }}
                      tickLine={false}
                      domain={[-4, 0]}
                      label={{
                        value: "Predicted log₁₀ VMR",
                        position: "bottom",
                        offset: 10,
                        style: { fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" },
                      }}
                    />
                    <YAxis
                      type="number"
                      dataKey="residual"
                      tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" }}
                      axisLine={{ stroke: "#d0d7de" }}
                      tickLine={false}
                      domain={[-0.6, 0.6]}
                      label={{
                        value: "Residual",
                        angle: -90,
                        position: "insideLeft",
                        offset: -5,
                        style: { fontSize: 11, fill: "#8b949e", fontFamily: "var(--font-plex-mono)" },
                      }}
                    />
                    <ZAxis type="number" dataKey="size" range={[15, 40]} />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          return (
                            <ChartTooltipContent
                              label={`Predicted: ${payload[0].payload.predicted.toFixed(2)}`}
                              value={payload[0].payload.residual.toFixed(3)}
                              unit="residual"
                            />
                          );
                        }
                        return null;
                      }}
                    />
                    <Scatter data={residualData} fill="#3b82f6" fillOpacity={0.35} stroke="#3b82f6" strokeOpacity={0.5} strokeWidth={0.5} />
                  </ScatterChart>
                </ResponsiveContainer>

                {/* Zero line annotation */}
                <div className="text-[11px] text-[#8b949e] text-center mt-1" style={{ fontFamily: "var(--font-plex-mono)" }}>
                  Horizontal center = zero bias
                </div>
              </div>
            </div>

            <CodeBlock
              code={`residuals = y_true - y_pred

Plot.plot({
  marks: [
    Plot.dot(holdout, {
      x: "predicted", y: "residual",
      fill: "#3b82f6", fillOpacity: 0.35,
      r: 2.5
    }),
    Plot.ruleY([0], { stroke: "#d0d7de", strokeDasharray: "4,3" })
  ],
  x: { label: "Predicted log₁₀ VMR" },
  y: { label: "Residual", domain: [-0.6, 0.6] }
})`}
              language="python"
            />
          </CellWrapper>

          <SectionDivider />

          {/* CONCLUSIONS */}
          <CellWrapper id="conclusions">
            <div className="group relative">
              <div className="absolute right-0 top-0">
                <PinButton />
              </div>
              <Prose>
                <h2 className="text-[20px] font-semibold text-[#1a1a2e] mb-1 tracking-[-0.01em]">
                  Key Findings
                </h2>
              </Prose>

              <div className="mt-4 space-y-3">
                {[
                  {
                    num: "01",
                    title: "Quantum advantage in feature extraction",
                    desc: "The parameterized quantum circuit captures higher-order correlations in spectral data that classical architectures miss, contributing to a 7.8% improvement over the ADC 2023 winner.",
                  },
                  {
                    num: "02",
                    title: "Robust across molecular targets",
                    desc: "Consistent sub-0.4 RMSE across all five molecules (H₂O, CO₂, CO, CH₄, NH₃), with particularly strong performance on water vapor (0.218).",
                  },
                  {
                    num: "03",
                    title: "NISQ-compatible design",
                    desc: "The 12-qubit, 8-layer circuit is executable on current hardware (IQM Spark, VTT Q50) with manageable noise levels — no error correction required.",
                  },
                  {
                    num: "04",
                    title: "First quantum ML for biosignature detection",
                    desc: "To our knowledge, ExoBiome is the first system to apply quantum machine learning to the detection and quantification of atmospheric biosignatures from transmission spectra.",
                  },
                ].map((item) => (
                  <div
                    key={item.num}
                    className="flex gap-4 p-4 rounded-lg border border-[#d8dee4] hover:border-[#3b82f6]/30 transition-colors"
                  >
                    <div
                      className="text-[13px] font-medium text-[#3b82f6] mt-0.5"
                      style={{ fontFamily: "var(--font-plex-mono)" }}
                    >
                      {item.num}
                    </div>
                    <div>
                      <div className="text-[14px] font-semibold text-[#1a1a2e]">{item.title}</div>
                      <div className="text-[13px] text-[#57606a] mt-1 leading-[1.6]">{item.desc}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Tech stack */}
              <div className="mt-8 p-5 bg-[#f6f8fa] border border-[#d8dee4] rounded-lg">
                <div className="text-[11px] font-semibold text-[#57606a] uppercase tracking-wider mb-3">
                  Stack
                </div>
                <div className="flex flex-wrap gap-2">
                  {[
                    "Python",
                    "PyTorch",
                    "Qiskit",
                    "qiskit-on-iqm",
                    "sQUlearn",
                    "scikit-learn",
                    "TauREx 3",
                    "SpectRes",
                  ].map((tech) => (
                    <span
                      key={tech}
                      className="text-[12px] px-2.5 py-1 bg-white border border-[#d0d7de] rounded-md text-[#24292f]"
                      style={{ fontFamily: "var(--font-plex-mono)" }}
                    >
                      {tech}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <CodeBlock
              code={`// This notebook is part of the ExoBiome project.
// HACK-4-SAGES 2026 · ETH Zurich (COPL)
// Category: "Life Detection and Biosignatures"
//
// Source: github.com/exobiome/quantum-biosignature-detection
// License: Apache 2.0`}
            />
          </CellWrapper>

          {/* FOOTER */}
          <div className="mt-16 pt-6 border-t border-[#d8dee4]">
            <div className="flex items-center justify-between text-[12px] text-[#8b949e]">
              <div className="flex items-center gap-2">
                <svg width="16" height="16" viewBox="0 0 22 22">
                  <circle cx="11" cy="11" r="9" stroke="#8b949e" strokeWidth="1.5" fill="none" />
                  <circle cx="11" cy="11" r="3" fill="#8b949e" />
                </svg>
                <span>Built with Observable</span>
              </div>
              <div className="flex items-center gap-4" style={{ fontFamily: "var(--font-plex-mono)" }}>
                <span>HACK-4-SAGES 2026</span>
                <span>·</span>
                <span>ETH Zurich</span>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* SCROLL-BASED TOC HIGHLIGHT OBSERVER */}
      <ScrollObserver onSectionChange={setActiveSection} />
    </div>
  );
}

function ScrollObserver({ onSectionChange }: { onSectionChange: (id: string) => void }) {
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            onSectionChange(entry.target.id);
          }
        }
      },
      { rootMargin: "-20% 0px -70% 0px" }
    );

    tocSections.forEach((section) => {
      const el = document.getElementById(section.id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [onSectionChange]);

  return null;
}
